import os
import sys
from argparse import ArgumentParser
from datetime import datetime
import csv
import re

import json
import yaml
import pandas as pd
from pandas.api.types import CategoricalDtype

from python_lib.errors import ExtensionError, MappingError
from python_lib.querying_hpds import get_HPDS_connection, query_runner, HpdsHTTPError, EmptyDataFrameError
from python_lib.quality_checking import quality_filtering
from python_lib.descriptive_statistics import get_descriptive_statistics
from python_lib.associative_statistics import associationStatistics, \
    EndogOrExogUnique, \
    CrossCountThresholdError, \
    nonNA_crosscount
from scipy.linalg import LinAlgError
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def upload_dropbox(function):
    def inner_function(*args, **kwargs):
        if args[0].parameters_exp["storage_dropbox"] is True:
            #TODO: to be implemented eventually
            pass
        return function(*args, **kwargs)
    return inner_function


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
        self.phs = phs
        self.batch_group = batch_group
        self.parameters_exp = parameters_exp
        self.results_path = os.path.join(parameters_exp["results_path"], time_launched)
        mapping_independent_variables = pd.read_csv("env_variables/df_eligible_variables.csv")\
            .loc[lambda df: (df["phs"] == phs) & (df["batch_group"] == batch_group), :]\
            .set_index("var_name", drop=True)["independent_var_id"]
        if self.parameters_exp["harmonized_variables_types"] == "categorical":
            mapping_dependent_variables = pd.read_csv("env_variables/df_harmonized_variables.csv") \
                                                 .loc[lambda df: df["categorical"] == True, :]\
                                                 .set_index("var_name", drop=True)["dependent_var_id"]
        elif self.parameters_exp["harmonized_variables_types"] == "all":
            mapping_dependent_variables = pd.read_csv("env_variables/df_harmonized_variables.csv") \
                .set_index("var_name", drop=True)["dependent_var_id"]
        else:
            raise ValueError("harmonized_variables_types should be either 'categorical' or 'all'")
        self.dependent_var_names = mapping_dependent_variables.index.tolist()
        self.independent_var_names = mapping_independent_variables.index.tolist()
        self.mapping_variables = mapping_independent_variables \
            .append(mapping_dependent_variables) \
            .rename("var_id")
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
    
    def var_name_to_id_df(self, df_to_map: pd.DataFrame, colnames) -> pd.DataFrame:
        df_mapped = df_to_map.replace({col: self.mapping_variables for col in colnames})
        if df_mapped[colnames].isna().any() is True:
            raise MappingError("Issue when mapping variables from names to ids")
        df_mapped = df_mapped.rename({col: re.sub("name", "id", col) for col in colnames}, axis=1)
        return df_mapped

    def var_id_to_name_df(self, df_to_map: pd.DataFrame, colnames):
        reverse_mapping = {v: k for k, v in self.mapping_variables.items()}
        df_mapped = df_to_map.replace({col: reverse_mapping for col in colnames})
        if df_mapped[colnames].isna().any() is True:
            raise MappingError("Issue when mapping variables from ids to names")
        df_mapped = df_mapped.rename({col: re.sub("id", "name", col) for col in colnames}, axis=1)
        return df_mapped

    @upload_dropbox
    def write_file(self, py_object, file_name, dir_path, map_colnames=False, colnames=None, *args, **kwargs):
        file_path = os.path.join(dir_path, file_name)
        extension_regex = re.compile(r"\.[a-z]+$")
        extension = re.search(extension_regex, file_path).group()
        if extension in [".csv", ".zip"]:
            if map_colnames is True:
                py_object = self.var_name_to_id_df(py_object, colnames)
            py_object.to_csv(file_path, *args, **kwargs)
        elif extension == ".json":
            with open(file_path, "w+") as json_file:
                json.dump(py_object, json_file, *args, **kwargs)
        elif extension == ".txt":
            with open(file_path, "w+") as text_file:
                text_file.write(py_object)
        else:
            raise ExtensionError
        
    @staticmethod
    def logs_to_df(dic_logs: dict) -> pd.DataFrame:
        logs_df = pd.DataFrame.from_dict({
            (dep, ind): logs for dep, dic_log_ind in dic_logs.items() for ind, logs in dic_log_ind.items()
        }, orient="index") \
            .rename_axis(["dependent_var_name", "independent_var_name"], axis=0) \
            .reset_index(drop=False)
        return logs_df
    
    def var_name_to_id_logs(self):
        pass
    
    def var_id_to_names_logs(self):
        pass
    
    
    def querying_data(self):
        path_log = os.path.join(self.results_path, "logs_hpds_query", self.phs)
        path_data = os.path.join("data", self.phs)
        if not os.path.isdir(path_log):
            os.makedirs(path_log)
        logs = {
            "phs": self.phs,
            "batch_group": self.batch_group,
            "time": datetime.now().strftime("%y/%m/%d %H:%M:%S")
        }
        try:
            if self.parameters_exp["online"] is True:
                resource = get_HPDS_connection(self.token,
                                               self.picsure_network_url,
                                               self.resource_id)
                self.study_df = query_runner(resource=resource,
                                             to_select=self.dependent_var_names,
                                             to_anyof=self.independent_var_names,
                                             result_type="DataFrame",
                                             low_memory=False,
                                             timeout=500) \
                    [self.dependent_var_names + self.independent_var_names]
            else:
                self.study_df = pd.read_csv(os.path.join(path_data, str(self.batch_group) + ".csv"))
            if self.parameters_exp["save"] is True:
                if not os.path.isdir(path_data):
                    os.makedirs(path_data)
                self.write_file(py_object=self.study_df,
                                file_name=str(self.batch_group) + ".csv",
                                dir_path=path_data,
                                index=False)
            
        except (EmptyDataFrameError, HpdsHTTPError) as exception:
            logs = RunPheWAS.logs_creating(exception)
        else:
            logs = RunPheWAS.logs_creating()
        finally:
            self.write_file(py_object=logs,
                            dir_path=path_log,
                            file_name=str(self.batch_group) + ".json")
        if logs["error"] is True:
            print(
                "Error during hpds data querying of {phs}, batch_group: {batch_group} \n{exception} \nQuitting program".format(
                    phs=self.phs,
                    batch_group=self.batch_group,
                    exception=repr(exception)))
            sys.exit()
        return
    
    def quality_checking(self):
        self.filtered_df = quality_filtering(self.study_df, self.parameters_exp["Minimum number observations"])
        self.filtered_independent_var_names = list(set(self.filtered_df.columns) - set(self.dependent_var_names))
        self.filtered_dependent_var_names = list(set(self.filtered_df.columns) - set(self.independent_var_names))
        results_path = os.path.join(self.results_path, "quality_checking", self.phs)
        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        # TODO: maybe adding the count of variables discarded because below threshold to get this information somewhere
        quality_checked = pd.DataFrame.from_dict({"var_name": self.study_df.columns}) \
            .assign(kept=lambda df: df["var_name"].isin(self.filtered_df.columns))
        self.write_file(quality_checked,
                        dir_path=results_path,
                        file_name=str(self.batch_group) + ".csv",
                        map_colnames=self.parameters_exp["var_name_to_id_df"],
                        colnames=["var_name"],
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
        dependent_var_types = pd.read_csv("env_variables/df_harmonized_variables.csv",
                                          index_col=0) \
            .loc[self.filtered_dependent_var_names, "var_type"] \
            .to_dict()
        self.var_types = {**dependent_var_types,
                          **independent_var_types
                          }
        
        results_path = os.path.join(self.results_path, "descriptive_statistics", self.phs)
        if not os.path.isdir(results_path):
            os.makedirs(results_path, exist_ok=True)
        self.write_file(self.descriptive_statistics_df,
                        file_name=str(self.batch_group) + ".csv",
                        dir_path=results_path,
                        map_colnames=self.parameters_exp["var_name_to_id_df"],
                        colnames=["var_name"],
                        index=False)
    
    def data_type_management(self):
        categorical_variables = [var_name for var_name in self.filtered_df.columns if
                                 self.var_types[var_name] != "continuous"]
        self.filtered_df[categorical_variables] = self.filtered_df[categorical_variables] \
            .apply(lambda serie: serie.astype(CategoricalDtype(ordered=False)))
    
    def association_statistics(self):
        dic_results_dependent_var = {}
        dic_logs_dependent_var = {}
        counter_dependent_var = 1
        for dependent_var_name in self.filtered_dependent_var_names:
            print(dependent_var_name)
            print("harmonized_var: {0} our of {1}".format(counter_dependent_var, len(self.filtered_dependent_var_names)))
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
                print(independent_var_name)
                print("{n} out of {total_var}".format(n=n, total_var=len(subset_independent_var_names)))
                independent_var = self.filtered_df[independent_var_name]
                association_statistics_instance = associationStatistics(dependent_var,
                                                                        independent_var)
                association_statistics_instance.drop_na()
                association_statistics_instance.normalize_continuous_var()
                association_statistics_instance.groupby_statistics()
                try:
                    association_statistics_instance.recode_levels()
                    association_statistics_instance.test_statistics()
                    model = association_statistics_instance.create_model()
                    results = model.fit()
                # todo: eventually remove this KeyError Exception
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
            counter_dependent_var += 1
        df_all_results = pd.concat(dic_results_dependent_var,
                                   axis=0,
                                   ignore_index=False) \
            .reset_index(level=0, drop=False) \
            .rename({"level_0": "dependent_var_name"}, axis=1) \
            .reset_index(drop=True)
        dir_results = os.path.join(self.results_path, "association_statistics", self.phs)
        if not os.path.isdir(dir_results):
            os.makedirs(dir_results, exist_ok=True)
        self.write_file(df_all_results,
                        file_name=str(self.batch_group) + ".csv",
                        dir_path=dir_results,
                        map_colnames=self.parameters_exp["var_name_to_id_df"],
                        colnames=["dependent_var_name", "independent_var_name"],
                        index=False)
        
        dir_logs = os.path.join(self.results_path, "logs_association_statistics", self.phs)
        if not os.path.isdir(dir_logs):
            os.makedirs(dir_logs, exist_ok=True) # exist_ok circumvent weird OS error in EC2 when process is launched by bash 
        df_logs_dependent_var = self.logs_to_df(dic_logs_dependent_var)
        self.write_file(df_logs_dependent_var,
                        file_name=str(self.batch_group) + ".csv",
                        dir_path=dir_logs,
                        map_colnames=self.parameters_exp["var_name_to_id_df"],
                        colnames=["dependent_var_name", "independent_var_name"])
        return


if __name__ == '__main__':
    from env_variables.env_variables import TOKEN, PICSURE_NETWORK_URL, RESOURCE_ID

    with open("env_variables/parameters_exp.yaml", "r") as f:
        parameters_exp = yaml.load(f, Loader=yaml.SafeLoader)

    start_time = datetime.now()
    parser = ArgumentParser()
    parser.add_argument("--phs", dest="phs", type=str, default=None)
    parser.add_argument("--batch_group", dest="batch_group", type=int, default=None)
    parser.add_argument("--time_launched", dest="time_launched", type=str, default="010101_000000")
    args = parser.parse_args()
    phs = args.phs
    batch_group = args.batch_group
    time_launched = args.time_launched
    monitor_file_name = os.path.join(parameters_exp["results_path"], time_launched, "monitor_process.csv")
    
    
    PheWAS = RunPheWAS(TOKEN,
                       PICSURE_NETWORK_URL,
                       RESOURCE_ID,
                       batch_group=batch_group,
                       phs=phs,
                       parameters_exp=parameters_exp,
                       time_launched=time_launched
                       )
    if True:
        PheWAS.independent_var_names = PheWAS.independent_var_names[0:100]
        PheWAS.dependent_var_names = \
            ["\\DCC Harmonized data set\\01 - Demographics\\Harmonized race category of participant.\\", 
             "\\DCC Harmonized data set\\01 - Demographics\\Indicator of Hispanic or Latino ethnicity.\\", 
             "\\DCC Harmonized data set\\01 - Demographics\\Subject sex  as recorded by the study.\\", 
             "\\DCC Harmonized data set\\01 - Demographics\\classification of Hispanic/Latino background for Hispanic/Latino subjects where country or region of origin information is available\\", 
             "\\DCC Harmonized data set\\02 - Atherosclerosis\\Common carotid intima-media thickness  calculated as the mean of two values: mean of multiple thickness estimates from the left far wall and from the right far wall.\\", 
             "\\DCC Harmonized data set\\02 - Atherosclerosis\\Coronary artery calcification (CAC) score using Agatston scoring of CT scan(s) of coronary arteries\\", 
             "\\DCC Harmonized data set\\02 - Atherosclerosis\\Extent of narrowing of the carotid artery.\\", 
             "\\DCC Harmonized data set\\02 - Atherosclerosis\\Presence or absence of carotid plaque.\\", 
             "\\DCC Harmonized data set\\03 - Baseline common covariates\\Body height at baseline.\\", 
             "\\DCC Harmonized data set\\03 - Baseline common covariates\\Body mass index calculated at baseline.\\", 
             "\\DCC Harmonized data set\\03 - Baseline common covariates\\Body weight at baseline.\\", 
             "\\DCC Harmonized data set\\03 - Baseline common covariates\\Indicates whether subject currently smokes cigarettes.\\", 
             "\\DCC Harmonized data set\\03 - Baseline common covariates\\Indicates whether subject ever regularly smoked cigarettes.\\", 
             "\\DCC Harmonized data set\\04 - Blood cell count\\Count by volume  or number concentration (ncnc)  of lymphocytes in the blood (bld).\\", 
             "\\DCC Harmonized data set\\04 - Blood cell count\\Count by volume  or number concentration (ncnc)  of neutrophils in the blood (bld).\\", 
             "\\DCC Harmonized data set\\04 - Blood cell count\\Count by volume  or number concentration (ncnc)  of platelets in the blood (bld).\\", 
             "\\DCC Harmonized data set\\04 - Blood cell count\\Count by volume  or number concentration (ncnc)  of red blood cells in the blood (bld).\\", 
             "\\DCC Harmonized data set\\04 - Blood cell count\\Count by volume  or number concentration (ncnc)  of white blood cells in the blood (bld).\\", 
             "\\DCC Harmonized data set\\04 - Blood cell count\\Measurement of hematocrit  the fraction of volume (vfr) of blood (bld) that is composed of red blood cells.\\", 
             "\\DCC Harmonized data set\\05 - Blood pressure\\Indicator for use of antihypertensive medication at the time of blood pressure measurement.\\", 
             "\\DCC Harmonized data set\\05 - Blood pressure\\Resting diastolic blood pressure from the upper arm in a clinical setting.\\", 
             "\\DCC Harmonized data set\\05 - Blood pressure\\Resting systolic blood pressure from the upper arm in a clinical setting.\\", 
             "\\DCC Harmonized data set\\06 - Lipids\\Blood mass concentration of high-density lipoprotein cholesterol\\", 
             "\\DCC Harmonized data set\\06 - Lipids\\Blood mass concentration of low-density lipoprotein cholesterol\\", 
             "\\DCC Harmonized data set\\06 - Lipids\\Blood mass concentration of total cholesterol\\", 
             "\\DCC Harmonized data set\\06 - Lipids\\Blood mass concentration of triglycerides\\", 
             "\\DCC Harmonized data set\\06 - Lipids\\Indicates whether participant was taking any lipid-lowering medication at blood draw to measure lipids phenotypes\\", 
             "\\DCC Harmonized data set\\07 - Venous Thromboembolism Event\\An indicator of whether a subject experienced a venous thromboembolism event (VTE) that was verified by adjudication or by medical professionals.\\", 
             "\\DCC Harmonized data set\\07 - Venous Thromboembolism Event\\An indicator of whether a subject had a venous thromboembolism (VTE) event prior to the start of the medical review process (including self-reported events).\\"
            ]

        # PheWAS.dependent_var_names =                    [
        #     "\\DCC Harmonized data set\\03 - Baseline common covariates\\Body height at baseline.\\",
        #      '\\DCC Harmonized data set\\02 - Atherosclerosis\\Extent of narrowing of the carotid artery.\\',
        #      "\\DCC Harmonized data set\\01 - Demographics\\Subject sex  as recorded by the study.\\",
        #      "\\DCC Harmonized data set\\01 - Demographics\\Harmonized race category of participant.\\",
        #      "\\DCC Harmonized data set\\04 - Blood cell count\\Measurement of the ratio of variation in width to the mean width of the red blood cell (rbc) volume distribution curve taken at +/- 1 CV  known as red cell distribution width (RDW).\\",
        #     "\\DCC Harmonized data set\\06 - Lipids\\Blood mass concentration of triglycerides\\"
        #     ]
        # PheWAS.dependent_var_names = ["\\DCC Harmonized data set\\04 - Blood cell count\\Measurement of the mass concentration (mcnc) of hemoglobin in a given volume of packed red blood cells (rbc)  known as mean corpuscular hemoglobin concentration (MCHC).\\"]
        # PheWAS.independent_var_names = ["\\Cardiovascular Health Study (CHS) Cohort: an NHLBI-funded observational study of risk factors for cardiovascular disease in adults 65 years or older ( phs000287 )\\Data contain extensive medical history information of subjects (all > 65 years of age)\\K-channel blockers to enhance insulin se\\"]
        # batch_group=4 phs=phs000287

    print(phs, batch_group)
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
    
    # Workaround because open(..., "a") not allowed on EC2 (OS Error 95), so reading csv file and appening row manually 
    end_time = datetime.now()
    new_row = pd.DataFrame([phs,
                            str(batch_group),
                            str(len(PheWAS.independent_var_names)),
                            str(len(PheWAS.filtered_independent_var_names)),
                            str(len(PheWAS.filtered_dependent_var_names)),
                            start_time.strftime("%y/%m/%d %H:%M:%S"),
                            end_time.strftime("%y/%m/%d %H:%M:%S"),
                            str(end_time - start_time).split(".")[0]
                           ]).transpose()
    if os.path.exists(monitor_file_name):
        monitor_df = pd.read_csv(monitor_file_name, header=None)
        monitor_df = monitor_df.append(new_row)
        monitor_df.to_csv(monitor_file_name, header=False, index=False)
    else: 
        new_row.to_csv(monitor_file_name, header=False, index=False)
#     with open(os.open(monitor_file_name, os.O_CREAT | os.O_WRONLY, 0o777), 'w') as tsvfile:
#         
#         writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
#         writer.writerow([phs, str(batch_group), start_time.strftime("%y/%m/%d %H:%M:%S"), end_time.strftime("%y/%m/%d %H:%M:%S"), str(end_time - start_time).split(".")[0]])
