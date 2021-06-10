import os
import sys
from argparse import ArgumentParser
from datetime import datetime
import csv

import json
import yaml
import pandas as pd
from pandas.api.types import CategoricalDtype

from python_lib.querying_hpds import get_HPDS_connection, query_runner, HpdsHTTPError, EmptyDataFrameError
from python_lib.quality_checking import quality_filtering
from python_lib.descriptive_statistics import get_descriptive_statistics
from python_lib.associative_statistics import associationStatistics, \
    EndogOrExogUnique, \
    CrossCountThresholdError, \
    nonNA_crosscount
from scipy.linalg import LinAlgError
from statsmodels.tools.sm_exceptions import PerfectSeparationError


class NoVariablesError(Exception):
    """
        Raised when all variables are being filtered by quality checking
    """


class MappingError(Exception):
    """
        Raised when variables mapping are
    """


def var_name_to_id(df_to_map: pd.DataFrame) -> pd.DataFrame:
    df_dependent_vars_id = pd.read_csv("env_variables/list_harmonized_variables.csv") \
        .set_index("var_name", drop=True)["dependent_var_id"].to_frame()
    df_independent_vars_id = pd.read_csv("env_variables/list_eligible_variables.csv") \
        .set_index("var_name", drop=True)["independent_var_id"].to_frame()
    df_mapped = df_to_map.join(df_independent_vars_id, how="left", on="independent_var_name") \
        .join(df_dependent_vars_id, how="left", on="dependent_var_name") \
        .drop(["dependent_var_name", "independent_var_name"], axis=1)
    if df_mapped[["dependent_var_id", "independent_var_id"]].isna().any() is True:
        raise MappingError("Issue when mapping variables from names to ids")
    return df_mapped


def var_id_to_name(df_to_map: pd.DataFrame):
    df_dependent_vars_id = pd.read_csv("env_variables/list_harmonized_variables.csv") \
        .set_index("var_name", drop=True)["dependent_var_id"].to_frame()
    df_independent_vars_id = pd.read_csv("env_variables/list_eligible_variables.csv") \
        .set_index("var_name", drop=True)["independent_var_id"].to_frame()
    df_mapped = df_to_map.join(df_independent_vars_id, how="left", on="independent_var_id") \
        .join(df_dependent_vars_id, how="left", on="dependent_var_id") \
        .drop(["dependent_var_id", "independent_var_id"], axis=1)
    if df_mapped[["dependent_var_name", "independent_var_name"]].isna().any() is True:
        raise MappingError("Issue when mapping variables from ids to names")
    return df_mapped


class RunPheWAS:
    def __init__(self,
                 token,
                 picsure_network_url,
                 resource_id,
                 batch_group,
                 phs,
                 parameters_exp,
                 time_launched):
        self.token = token
        self.picsure_network_url = picsure_network_url
        self.resource_id = resource_id
        self.resource = None
        self.phs = phs
        self.batch_group = batch_group
        self.parameters_exp = parameters_exp
        self.results_path = os.path.join("results", time_launched)
        eligible_variables = pd.read_csv("env_variables/list_eligible_variables.csv")
        list_harmonized_variables = pd.read_csv("env_variables/list_harmonized_variables.csv")
        if self.parameters_exp["harmonized_variables_types"] == "categorical":
            self.dependent_var_names = list_harmonized_variables.loc[
                lambda df: df["categorical"] == True, "var_name"].tolist()
        elif self.parameters_exp["harmonized_variables_types"] == "all":
            self.dependent_var_names = list_harmonized_variables.loc[:, "var_name"].tolist()
        else:
            raise ValueError("harmonized_variables_types should be either 'categorical' or 'all'")
        self.independent_var_names = eligible_variables \
            .loc[lambda df: (df["phs"] == phs) & (df["batch_group"] == batch_group), "var_name"] \
            .tolist()
        self.var_types = None
        self.study_df = None
        self.filtered_df = None
        self.filtered_independent_var_names = None
        self.filtered_dependent_var_names = None
        self.descriptive_statistics_df = None
        self.crosscount_filtered_independent_var_names = None
    
    @staticmethod
    def logs_creating(exception=None):
        error = False if exception is None else True
        if error is True:
            e = repr(exception)
        else:
            e = None
        logs = {
            "error": error,
            "logs": e
        }
        return logs
    
    def querying_data(self):
        path_log = os.path.join(self.results_path, "logs_hpds_query", self.phs)
        if not os.path.isdir(path_log):
            os.makedirs(path_log)
        logs = {
            "phs": self.phs,
            "batch_group": self.batch_group,
            "time": datetime.now().strftime("%y/%m/%d %H:%M:%S")
        }
        try:
            self.resource = get_HPDS_connection(self.token,
                                                self.picsure_network_url,
                                                self.resource_id)
            self.study_df = query_runner(resource=self.resource,
                                         to_select=self.dependent_var_names,
                                         to_anyof=self.independent_var_names,
                                         result_type="DataFrame",
                                         low_memory=False,
                                         timeout=500) \
                [self.dependent_var_names + self.independent_var_names]
        except (EmptyDataFrameError, HpdsHTTPError) as exception:
            logs["error"] = True
            e = repr(exception)
            logs["logs"] = e
        else:
            logs["error"] = False
            e = None
            logs["logs"] = e
        finally:
            with open(os.path.join(path_log, str(self.batch_group) + ".json"), "w+") as log_file:
                json.dump(logs, log_file)
        if logs["error"] is True:
            print(
                "Error during hpds data querying of {phs}, batch_group: {batch_group} \n{exception} \nQuitting program".format(
                    phs=self.phs,
                    batch_group=self.batch_group,
                    exception=e))
            sys.exit()
        return
    
    def quality_checking(self):
        self.filtered_df = quality_filtering(self.study_df, self.parameters_exp["Minimum number observations"])
        self.filtered_independent_var_names = list(set(self.filtered_df.columns) - set(self.dependent_var_names))
        self.filtered_dependent_var_names = list(set(self.filtered_df.columns) - set(self.independent_var_names))
        path_results = os.path.join(self.results_path, "quality_checking", self.phs)
        if not os.path.isdir(path_results):
            os.makedirs(path_results)
        # TODO: maybe adding the count of variables discarded because below threshold to get this information somewhere
        pd.DataFrame.from_dict({"variable_name": self.study_df.columns}) \
            .assign(kept=lambda df: df["variable_name"].isin(self.filtered_df.columns)) \
            .to_csv(os.path.join(path_results, str(self.batch_group) + ".csv"),
                    index=False)
        return
    
    def descriptive_statistics(self):
        
        self.descriptive_statistics_df = get_descriptive_statistics(
            self.filtered_df,
            self.filtered_independent_var_names
        )
        independent_var_types = self.descriptive_statistics_df[["var_name", "var_type"]] \
            .drop_duplicates() \
            .set_index("var_name")["var_type"] \
            .to_dict()
        dependent_var_types = pd.read_csv("env_variables/list_harmonized_variables.csv",
                                          index_col=0) \
            .loc[self.filtered_dependent_var_names, "var_type"] \
            .to_dict()
        self.var_types = {**dependent_var_types,
                          **independent_var_types
                          }
        
        path_results = os.path.join(self.results_path, "descriptive_statistics", self.phs)
        if not os.path.isdir(path_results):
            os.makedirs(path_results)
        self.descriptive_statistics_df.to_csv(os.path.join(path_results, str(self.batch_group) + ".csv"),
                                              index=False)
    
    def data_type_management(self):
        categorical_variables = [var_name for var_name in self.filtered_df.columns if
                                 self.var_types[var_name] != "continuous"]
        self.filtered_df[categorical_variables] = self.filtered_df[categorical_variables] \
            .apply(lambda serie: serie.astype(CategoricalDtype(ordered=False)))
    
    def association_statistics(self):
        dic_results_dependent_var = {}
        dic_logs_dependent_var = {}
        for dependent_var_name in self.filtered_dependent_var_names:
            print(dependent_var_name)
            dependent_var = self.filtered_df[dependent_var_name]
            dic_results_independent_var = {}
            dic_logs_independent_var = {}
            non_na_crosscount = nonNA_crosscount(self.filtered_df,
                                                 dependent_var_name,
                                                 self.filtered_independent_var_names)
            discarded_independent_variables = non_na_crosscount.loc[
                                              lambda df: df["value"] <= self.parameters_exp["threshold_crosscount"],
                                              :]
            for independent_var_name, statistics in discarded_independent_variables.iterrows():
                dic_results_independent_var[independent_var_name] = statistics.to_frame().transpose()
                try:
                    raise CrossCountThresholdError("Crosscount below {threshold}".format(
                        threshold=self.parameters_exp["threshold_crosscount"]))
                except CrossCountThresholdError as e:
                    dic_logs_independent_var[independent_var_name] = self.logs_creating(e)
            subset_independent_var_names = non_na_crosscount.index.drop(discarded_independent_variables.index)
            n = 0
            for independent_var_name in subset_independent_var_names:
                n += 1
                print("{n} out of {total_var}".format(n=n, total_var=len(subset_independent_var_names)))
                independent_var = self.filtered_df[independent_var_name]
                association_statistics_instance = associationStatistics(dependent_var,
                                                                        independent_var)
                try:
                    association_statistics_instance.drop_na()
                    association_statistics_instance.normalize_continuous_var()
                    association_statistics_instance.groupby_statistics()
                    association_statistics_instance.recode_levels()
                    association_statistics_instance.test_statistics()
                    model = association_statistics_instance.create_model()
                    results = model.fit()
                except (LinAlgError, PerfectSeparationError, EndogOrExogUnique, KeyError) as exception:
                    dic_logs_independent_var[independent_var_name] = self.logs_creating(exception)
                    association_statistics_instance.creating_empty_df_results()
                else:
                    dic_logs_independent_var[independent_var_name] = self.logs_creating()
                    association_statistics_instance.model_results_handling(results)
                finally:
                    dic_results_independent_var[
                        independent_var_name] = association_statistics_instance.gathering_statistics()
            dic_results_dependent_var[dependent_var_name] = pd.concat(
                dic_results_independent_var,
                axis=0,
                ignore_index=False) \
                .reset_index(drop=False, level=0) \
                .rename({"level_0": "independent_var_name"}, axis=1)
            dic_logs_dependent_var[dependent_var_name] = dic_logs_independent_var
        df_all_results = pd.concat(dic_results_dependent_var,
                                   axis=0,
                                   ignore_index=False) \
            .reset_index(level=0, drop=False) \
            .rename({"level_0": "dependent_var_name"}, axis=1) \
            .reset_index(drop=True)
        dir_results = os.path.join(self.results_path, "association_statistics", self.phs)
        if not os.path.isdir(dir_results):
            os.makedirs(dir_results)
        df_all_results.pipe(var_name_to_id) \
            .to_csv(os.path.join(dir_results, str(self.batch_group) + ".csv"),
                    index=False)
        dir_logs = os.path.join(self.results_path, "logs_association_statistics", self.phs)
        if not os.path.isdir(dir_logs):
            os.makedirs(dir_logs)
        path_logs = os.path.join(dir_logs, str(self.batch_group) + ".json")
        with open(path_logs, "w+") as f:
            json.dump(dic_logs_dependent_var, f)
        return


if __name__ == '__main__':
    from env_variables.env_variables import TOKEN, PICSURE_NETWORK_URL, RESOURCE_ID
    
    start_time = datetime.now().strftime("%y/%m/%d_%H:%M:%S")
    parser = ArgumentParser()
    parser.add_argument("--phs", dest="phs", type=str, default=None)
    parser.add_argument("--batch_group", dest="batch_group", type=int, default=None)
    parser.add_argument("--time_launched", dest="time_launched", type=str, default="010101_000000")
    args = parser.parse_args()
    phs = args.phs
    batch_group = args.batch_group
    time_launched = args.time_launched
    monitor_file_name = os.path.join("results", time_launched, "monitor_process.tsv")
    
    with open("env_variables/parameters_exp.yaml", "r") as f:
        parameters_exp = yaml.load(f, Loader=yaml.SafeLoader)
    
    PheWAS = RunPheWAS(TOKEN,
                       PICSURE_NETWORK_URL,
                       RESOURCE_ID,
                       batch_group=batch_group,
                       phs=phs,
                       parameters_exp=parameters_exp,
                       time_launched=time_launched
                       )
    if False:
        PheWAS.dependent_var_names = [
            "\\DCC Harmonized data set\\03 - Baseline common covariates\\Body height at baseline.\\",
            '\\DCC Harmonized data set\\02 - Atherosclerosis\\Extent of narrowing of the carotid artery.\\',
            "\\DCC Harmonized data set\\01 - Demographics\\Subject sex  as recorded by the study.\\",
            "\\DCC Harmonized data set\\01 - Demographics\\Harmonized race category of participant.\\"
        ]
        PheWAS.independent_var_names = PheWAS.independent_var_names[0:10]
    print("querying the data", datetime.now().time().strftime("%H:%M:%S"))
    PheWAS.querying_data()
    print("quality checking", datetime.now().time().strftime("%H:%M:%S"))
    PheWAS.quality_checking()
    if (len(PheWAS.filtered_independent_var_names) == 0) | (len(PheWAS.filtered_dependent_var_names) == 0):
        print("No valid variables, skipping")
    else:
        print("descriptive statistics", datetime.now().time().strftime("%H:%M:%S"))
        PheWAS.descriptive_statistics()
        print("data management", datetime.now().time().strftime("%H:%M:%S"))
        PheWAS.data_type_management()
        print("association statistics", datetime.now().time().strftime("%H:%M:%S"))
        PheWAS.association_statistics()
    
    with open(monitor_file_name, "a") as tsvfile:
        end_time = datetime.now().strftime("%y/%m/%d_%H:%M:%S")
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        writer.writerow([phs, str(batch_group), str(start_time), str(end_time)])
