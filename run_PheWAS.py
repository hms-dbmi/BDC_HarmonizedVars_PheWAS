import os
from argparse import ArgumentParser
import contextlib

import json
import yaml
import pandas as pd

from python_lib.querying_hpds import get_HPDS_connection, query_runner
from python_lib.quality_checking import quality_filtering
from python_lib.PheWAS_funcs import PheWAS


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
        self.filtered_var_names = None

    def querying_data(self):
        path_log = os.path.join("results/log/", self.phs, str(self.batch_group))
        if not os.path.isdir(path_log):
            os.makedirs(path_log)
        
        with open(os.path.join(path_log, "hpds_output.txt"), "w+") as file:
            with contextlib.redirect_stdout(f):
                study_df = query_runner(resource=self.resource,
                                        to_select=self.dependent_var_names,
                                        to_anyof=self.independent_var_names,
                                        result_type="DataFrame",
                                        low_memory=False,
                                        timeout=500)
                if len(study_df.columns) == 4 | study_df.shape[0] == 0:
                    raise ValueError("Data Frame empty, either no matching variable names, or server error")
        return study_df
    
    def quality_checking(self, study_df):
        self.filtered_df = quality_filtering(study_df, self.parameters_exp["Minimum number observations"])
        self.filtered_var_names = self.filtered_df.columns
        path_results = os.path.join("results/quality_checking", self.phs, str(self.batch_group))
        if not os.path.isdir(path_results):
            os.makedirs(path_results)
        pd.DataFrame.from_dict({"variable_name": study_df.columns})\
            .assign(kept=lambda df: df["variable_name"].isin(self.filtered_df.columns))\
            .to_csv(os.path.join(path_results, "quality_checking.csv"),
                    index=False)
        return
    
    def descriptive_statistics(self):
        #TODO:
        # - filtrer value_counts pour n'utiliser que les variables qualitatives
        # - rajouter total number of non null observations
        # - mean et median pour les variables continues
        # - pull it together
        # - rajouter self devant filtered_df et filtered_var_names
        # - eventuellement transferer les fonctions dans le fichier descriptive_statistics.py
        
        non_null_values = filtered_df.notnull().sum()
        
        def _count_modalities(var_name, serie):
            return serie.value_counts()\
                        .rename("value").to_frame()\
                        .rename_axis("modality", axis=0).reset_index(drop=False)\
                        .assign(var_name=var_name, categorical=True)
        
        counts = [_count_modalities(var_name, serie) for var_name, serie in filtered_df[filtered_var_names].iteritems()]
        
        categorical_value_counts = pd.concat(counts, ignore_index=True).assign(statistics="count")
        
        return
    
    def run_PheWAS(self):
        
        return


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
    
    run_PheWAS = RunPheWAS(TOKEN,
                           PICSURE_NETWORK_URL,
                           RESOURCE_ID,
                           batch_group=batch_group,
                           phs=phs,
                           parameters_exp=parameters_exp
                           )
    study_df = run_PheWAS.querying_data()
    run_PheWAS.quality_checking(study_df)
    run_PheWAS.descriptive_statistics()
    
    
    dependent_dic_pvalues = {}
    dependent_dic_errors = {}
    
    for dependent_var_name in dependent_var_names:
        print(phs)
        print(dependent_var_name)
        print("entering PheWAS")
        substudy_df = study_df[[dependent_var_name] + study_variables]
        dic_pvalues, dic_errors = PheWAS(substudy_df, dependent_var_name)
        print("PheWAS done")
        dependent_dic_pvalues[dependent_var_name] = dic_pvalues
        dependent_dic_errors[dependent_var_name] = dic_errors
    
    with open(os.path.join("results", "associations", phs + "_pvalues.json"), "w+") as f:
        json.dump(dependent_dic_pvalues, f)
    
    with open(os.path.join("results", "associations", phs + "_errors.json"), "w+") as f:
        json.dump(dependent_dic_errors, f)
