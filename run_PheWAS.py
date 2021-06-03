import os
from argparse import ArgumentParser
import contextlib
from datetime import datetime

import json
import yaml
import pandas as pd
from pandas.api.types import CategoricalDtype

from python_lib.querying_hpds import get_HPDS_connection, query_runner
from python_lib.quality_checking import quality_filtering
from python_lib.descriptive_statistics import get_descriptive_statistics
from python_lib.associative_statistics import associationStatistics
from scipy.linalg import LinAlgError
from statsmodels.tools.sm_exceptions import PerfectSeparationError


class RunPheWAS:
    
    def __init__(self,
                 token,
                 picsure_network_url,
                 resource_id,
                 batch_group,
                 phs,
                 parameters_exp):
        self.resource = get_HPDS_connection(token,
                                            picsure_network_url,
                                            resource_id)
        self.phs = phs
        self.batch_group = batch_group
        self.parameters_exp = parameters_exp
        # self.study_df = None
        self.filtered_df = None
        eligible_variables = pd.read_csv("env_variables/list_eligible_variables.csv")
        list_harmonized_variables = pd.read_csv("env_variables/list_harmonized_variables.csv")
        if self.parameters_exp["variable_types"] is True:
            self.dependent_var_names = list_harmonized_variables.loc[lambda df: df["categorical"] == True, "name"].tolist()
        else:
            self.dependent_var_names = list_harmonized_variables.loc[:, "name"].tolist()

        self.independent_var_names = eligible_variables.loc[lambda df: (df["phs"] == phs) & \
                                                                       (df["batch_group"] == batch_group),
                                                            "name"]
        self.var_types = list_harmonized_variables.set_index("name")["var_type"].drop_duplicates().to_dict()
        self.study_df = None
        self.filtered_independent_var_names = None
        self.descriptive_statistics_df = None

    def querying_data(self):
        path_log = os.path.join("results/log/", self.phs, str(self.batch_group))
        if not os.path.isdir(path_log):
            os.makedirs(path_log)
        
        with open(os.path.join(path_log, "hpds_output.txt"), "w+") as file:
            with contextlib.redirect_stdout(f):
                self.study_df = query_runner(resource=self.resource,
                                        to_select=self.dependent_var_names,
                                        to_anyof=self.independent_var_names,
                                        result_type="DataFrame",
                                        low_memory=False,
                                        timeout=500)
                if len(self.study_df.columns) == 4 | self.study_df.shape[0] == 0:
                    raise ValueError("Data Frame empty, either no matching variable names, or server error")
        return
    
    def quality_checking(self, study_df):
        self.filtered_df = quality_filtering(study_df, self.parameters_exp["Minimum number observations"])
        self.filtered_independent_var_names = list(set(self.filtered_df.columns) - set(self.dependent_var_names))
        path_results = os.path.join("results/quality_checking", self.phs)
        if not os.path.isdir(path_results):
            os.makedirs(path_results)
        pd.DataFrame.from_dict({"variable_name": study_df.columns})\
            .assign(kept=lambda df: df["variable_name"].isin(self.filtered_df.columns))\
            .to_csv(os.path.join(path_results, str(self.batch_group) + ".csv"),
                    index=False)
        return
    
    def descriptive_statistics(self):
        self.descriptive_statistics_df = get_descriptive_statistics(self.filtered_df, self.filtered_independent_var_names)
        self.var_types = {**self.var_types,
                          **PheWAS.descriptive_statistics_df.set_index("var_name")["var_type"].drop_duplicates().to_dict()
        }

        path_results = os.path.join("results/descriptive_statistics", self.phs)
        if not os.path.isdir(path_results):
            os.makedirs(path_results)
        self.descriptive_statistics_df.to_csv(os.path.join(path_results, str(self.batch_group) + ".csv"),
                                         index=False)
        return
    
    def association_statistics(self):
        dic_results_dependent_var = {}
        dic_logs_dependent_var = {}
        for dependent_var_name in self.dependent_var_names:
            dependent_var = self.study_df[dependent_var_name].astype(CategoricalDtype(ordered=False))
            dic_results_independent_var = {}
            dic_logs_independent_var = {}
            for independent_var_name in self.independent_var_names:
                independent_var = self.filtered_df[independent_var_name].astype(CategoricalDtype(ordered=False))
                association_statistics = associationStatistics(dependent_var, independent_var)
                model = association_statistics.create_model()
                try:
                    model.fit()
                    results = model.results
                except (LinAlgError, PerfectSeparationError) as exception:
                    dic_logs_independent_var[independent_var_name] = association_statistics.logs_creating(exception)
                    association_statistics.creating_empty_df_results()
                else:
                    dic_logs_independent_var[independent_var_name] = association_statistics.logs_creating()
                    association_statistics.model_results_handling(results)
                finally:
                    dic_results_independent_var[independent_var_name] = association_statistics.gathering_statistics()
            dic_results_dependent_var[dependent_var_name] = pd.concat(dic_results_independent_var,
                                                                      axis=0,
                                                                      ignore_index=True)
            dic_logs_dependent_var[dependent_var_name] = dic_logs_independent_var
        df_all_results = pd.concat(dic_results_dependent_var,
                                axis=0,
                                ignore_index=True)
        
        path_results = os.path.join("results/association_statistics", self.phs)
        if not os.path.isdir(path_results):
            os.makedirs(path_results)
        df_all_results.to_csv(os.path.join(path_results, str(self.batch_group) + ".csv"),
                              index=False)
        dir_logs = os.path.join("results/logs_association_statistics", self.phs)
        if not os.path.isdir(dir_logs):
            os.makedirs(dir_logs)
        path_logs = os.path.join(dir_logs, str(self.batch_group) + ".pickle")
        with open(path_logs, "w+") as f:
            json.dump(dic_logs_dependent_var, f)
        
        return df_all_results, dic_logs_dependent_var



if __name__ == '__main__':
    from env_variables.env_variables import TOKEN, PICSURE_NETWORK_URL, RESOURCE_ID
    
    parser = ArgumentParser()
    parser.add_argument("--phs", dest="phs", type=str, default=None)
    parser.add_argument("--batch_group", dest="batch_group", type=int, default=None)
    args = parser.parse_args()
    phs = args.phs
    batch_group = args.batch_group
    
    with open("env_variables/parameters_exp.yaml", "r") as f:
        parameters_exp = yaml.load(f, Loader=yaml.SafeLoader)
    
    print("creation of the object")
    PheWAS = RunPheWAS(TOKEN,
                       PICSURE_NETWORK_URL,
                       RESOURCE_ID,
                       batch_group=batch_group,
                       phs=phs,
                       parameters_exp=parameters_exp
                       )
    print("querying the data", datetime.now().time())
    PheWAS.querying_data()
    print("quality checking", datetime.now().time())
    PheWAS.quality_checking(PheWAS.study_df)
    print("descriptive statistics", datetime.now().time())
    PheWAS.descriptive_statistics()
    print("association statistics", datetime.now().time())
    df_all_results, dic_logs_dependent_var = PheWAS.association_statistics()
