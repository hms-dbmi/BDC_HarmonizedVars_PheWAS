import yaml
import os 
import json

import pandas as pd
import numpy as np


pd.set_option("display.max.rows", 100)

with open("env_variables/parameters_exp.yaml", "r") as f:
    parameters_exp = yaml.load(f, Loader=yaml.SafeLoader)
date_results = "211029_024422"
path_results = os.path.join(parameters_exp["results_path"], date_results)
path_monitor = os.path.join(path_results, "monitor_process.csv")

# Variables name dictionary 
dict_harmonized_variables = pd.read_csv("env_variables/df_harmonized_variables.csv")
multiindex_variablesDict = pd.read_csv("env_variables/multiIndex_variablesDict.csv", 
                                     low_memory=False)
dict_independent_variables = pd.read_csv("env_variables/df_eligible_variables.csv")\
                               .join(multiindex_variablesDict.set_index("name")["simplified_name"], 
                                    on="var_name")
                        

# Monitoring_table
monitoring_table = pd.read_csv(path_monitor, header=None, index_col = None)

if True: 
    monitoring_table = monitoring_table.sample(15)

monitoring_table_colnames = [
    "phs",
    "batch_group",
    "nb_independent_vars", 
    "nb_filtered_independent_vars", 
    "nb_dependent_vars", 
    "time_start", 
    "time_stop",
    "duration"
]
monitoring_table = monitoring_table.set_axis(monitoring_table_colnames, axis=1)\
                                   .sort_values(["phs", "batch_group"], ascending=True, ignore_index=True)


## All results 
all_association = []
all_descriptive = []
all_logs_association = []
for key, (phs, batch_group) in monitoring_table[["phs", "batch_group"]].iterrows():
    batch_group = str(batch_group)
    print(key) 
    print(batch_group)
    path_association = os.path.join(path_results, "association_statistics", phs, batch_group + ".csv")
    path_descriptive = os.path.join(path_results, "descriptive_statistics", phs, batch_group + ".csv")
    path_logs_association_statistics = os.path.join(
        path_results,
        "logs_association_statistics",
        phs,
        batch_group + ".csv"
    )
    all_association.append(pd.read_csv(path_association, 
                                       dtype = {
                                           "dependent_var_id": str, 
                                           "independent_var_id": str, 
                                           "value": float, 
                                           "indicator": str, 
                                           "dependent_var_modality": str, 
                                           "independent_var_modality": str, 
                                           "ref_modality_dependent": str, 
                                           "ref_modality_independent": str, 
                                       }
                                      ))
    all_descriptive.append(pd.read_csv(path_descriptive, 
                                       dtype = {
                                           "modality": str, 
                                           "value": float, 
                                           "var_id": str, 
                                           "var_type": str, 
                                           "statistic": str
                                       }
                                      ))
    all_logs_association.append(pd.read_csv(path_logs_association_statistics))

all_association_table = pd.concat(all_association)
all_descriptive_table = pd.concat(all_descriptive)
all_logs_association_table = pd.concat(all_logs_association)

# Sample results    
row_exp = 2
phs = monitoring_table["phs"].iloc[row_exp]
batch_group = str(monitoring_table["batch_group"].iloc[row_exp])

path_association = os.path.join(path_results, "association_statistics", phs, batch_group + ".csv")
path_descriptive = os.path.join(path_results, "descriptive_statistics", phs, batch_group + ".csv")
path_logs_association_statistics = os.path.join(path_results,
                                                "logs_association_statistics", phs, batch_group + ".csv")
path_logs_query = os.path.join(path_results, "logs_hpds_query", phs, batch_group + ".json")

        
# Loading results
association_table = pd.read_csv(path_association, low_memory=False)
descriptive_table = pd.read_csv(path_descriptive, low_memory=False)
logs_association_table = pd.read_csv(path_logs_association_statistics)
with open(path_logs_query, "r") as json_file:
    logs_query = json.load(json_file)
    

# Results table 
## read one results table
## filter by columns and pvalue significance
## gather results together
## output some human readable results
## present them in a way to get some insights about it 
n_tests_ran = association_table[["dependent_var_id", "independent_var_id"]].drop_duplicates().shape[0]

association_table.indicator.unique()
pvalues_name = ["pvalue_LRT_LR", "pvalue_LRT_LogR"]
significant_associations = association_table.loc[lambda df: df["indicator"].isin(pvalues_name), :]\
                 .loc[lambda df: df["value"] < 0.05/n_tests_ran, :]

sign_association_table = association_table.set_index(["dependent_var_id", "independent_var_id"])\
                                          .join(significant_associations.set_index(["dependent_var_id", "independent_var_id"])\
                                                .rename({"value": "significance"}, 
                                                        axis=1
                                                       )\
                                                .loc[:, "significance"], 
                                                   how="inner", 
                                                   rsuffix="_r"
                                               )\
                                          .reset_index(drop=False)

sign_association_table = sign_association_table.join(dict_independent_variables\
                                                     .set_index("independent_var_id")\
                                                     [["var_name", "simplified_name"]], 
                            on="independent_var_id")\
                      .rename({"var_name": "independent_var_name", 
                              "simplified_name": "simplified_independent_var_name"}, axis=1)\
                      .join(dict_harmonized_variables.set_index("dependent_var_id")["renaming_variables_nice"], 
                            on="dependent_var_id")\
                      .rename({"renaming_variables_nice": "dependent_var_name"}, axis=1)

sign_association_table.loc[lambda df: df["indicator"].isin(["coeff_logR", "coeff_LR"]), ]\
                      .assign(abs_value = lambda df: np.abs(df["value"]))\
                      .sort_values("abs_value", ascending=False)\
                      .head(100)

#TODO: create a visualization from these results 


# Getting Flow Chart information 

# Inputs : 
monitoring_table
association_table
descriptive_table
logs_association_table

# Gather all the association tables and 
logs_association_table_all

# Step 1: nb total independent_vars
step1 = monitoring_table["nb_independent_vars"].sum()
monitoring_table["nb_dependent_vars"].max()
# Step 2: discarding at the descriptive part: not enough non null values in the variable
step2 = monitoring_table["nb_filtered_independent_vars"].sum()
# Step 2bis: 
step2bis = all_logs_association_table[["dependent_var_id", "independent_var_id"]]\
                                 .drop_duplicates()\
                                 .shape[0]

# Step 3: How many remaining after crosscount threshold qc
step3 = all_logs_association_table.loc[lambda df: df["logs"] != "CrossCountThresholdError('Crosscount below 10',)",:]
step3[["dependent_var_id", "independent_var_id"]]\
                      .drop_duplicates()\
                      .shape[0]
# Step 4: How many remaining after statistical issue
step4 = step3.loc[lambda df: ~ df["error"] != True,:]
step4[["dependent_var_id", "independent_var_id"]]\
                      .drop_duplicates()\
                      .shape[0]

# Step 5: How many ran
all_association_table.loc[lambda df: df["indicator"].isin(["pvalue_LRT_LogR", "pvalue_LRT_LogR"]), ["dependent_var_id", "independent_var_id"]]\
    .drop_duplicates()\
    .shape[0]

step5 = all_association_table.loc[lambda df: df["value"].notnull() &
                          df["indicator"].isin(["pvalue_LRT_LogR", "pvalue_LRT_LogR"]) &
                          (df["dependent_var_modality"].isna() |
                          (df["dependent_var_modality"] == "overall_margin")) &
                          (df["independent_var_modality"].isna() |
                          (df["independent_var_modality"] == "overall_margin")),:]

step5[["dependent_var_id", "independent_var_id"]]\
    .drop_duplicates()\
    .shape[0]


# significant_associations
bonferonni_threshold = 0.05 / step3.shape[0]

significant_05 = step5.loc[lambda df: df["value"] < 0.05, :]
significant_adjusted = step5.loc[lambda df: df["value"] < bonferonni_threshold, :]


# Step 4: How many potential (looking only at overall margin)
step3_potential = logs_association_table[["dependent_var_id", "independent_var_id"]].drop_duplicates() # 2619 rows
step3_performed_wo_error = logs_association_table.loc[lambda df: df["logs"].isna(), :]\
                                                 .loc[:, ["dependent_var_id", "independent_var_id"]]\
                                                 .drop_duplicates()\
                                                 .shape[0] # 1711 rows



# Comprendre mismatch entre association_table et logs_association_table
association_table.loc[lambda df: ((df["dependent_var_modality"] == "overall_margin") | df["dependent_var_modality"].isna()) |
                                  ((df["independent_var_modality"] == "overall_margin") | df["independent_var_modality"].isna()), :]\
[["dependent_var_id", "independent_var_id"]].drop_duplicates().shape # 2619 rows

subset = association_table.loc[lambda df: ((df["dependent_var_modality"] == "overall_margin") | df["dependent_var_modality"].isna()) |
                                  ((df["independent_var_modality"] == "overall_margin") | df["independent_var_modality"].isna()), :]

subset[["dependent_var_id", "independent_var_id"]].drop_duplicates().shape # 2619 rows

subset_association_table = subset.loc[lambda df: df["indicator"].isin(["coeff_LogR", "coeff_LR"]), :]\
                                 .drop_duplicates(["dependent_var_id", "independent_var_id"]).shape # 1791 rows

subset_association_table = association_table.loc[lambda df: df["indicator"].isin(["coeff_LogR", "coeff_LR"]) &
                                                            df["value"].notnull(), :]\
                                   .drop_duplicates(["dependent_var_id", "independent_var_id"]) # 1694 rows
# Comprendre

subset_association_table["value"].isna().value_counts()

subset_log_association = logs_association_table.loc[lambda df: df["logs"].isna(), :]

subset_association_table.join(subset_log_association.set_index(["dependent_var_id", "independent_var_id"])["error"], 
                             on=["dependent_var_id", "independent_var_id"], 
                             how="left")\
                        .loc[lambda df: df["error"] == True,:]

step_3_completed = association_table[]
step_3_succeed = association_table
logs_association_table.loc[lambda df: (df["dependent_var_id"] == "D39") & (df["independent_var_id"] == "I16555"), :]


# Reading HPDS logs 


path_log_hpds = "logs_hpds_query"
list_phs_batch_group = pd.read_csv("/home/ec2-user/SageMaker/BDC_HarmonizedVars_PheWAS/env_variables/list_phs_batchgroup.csv")
import json
import os

list_phs = list_phs_batch_group["phs"].unique()
dic_logs_hpds = {}
for phs in list_phs:
    print(phs)
    dic_logs_hpds[phs] = {}
    list_batches = list_phs_batch_group.loc[lambda df: df["phs"] == phs, "batch_group"].values.astype(str).tolist()
    for batch in list_batches:
        print(batch)
        try:
            with open(os.path.join(path_results, path_log_hpds, phs, batch + ".json"), "r") as f: 
                log_dict = dic_logs_hpds[batch] = json.load(f)
            dic_logs_hpds[phs][batch] = log_dict
        except FileNotFoundError:
            dic_logs_hpds[phs][batch] = "File not found"


df = pd.DataFrame.from_dict({(phs, batch): dic_logs_hpds[phs][batch]
                            for phs in dic_logs_hpds.keys()
                            for batch in dic_logs_hpds[phs].keys()},
                            orient="index")

series = pd.Series({(phs, batch): dic_logs_hpds[phs][batch]
                            for phs in dic_logs_hpds.keys()
                            for batch in dic_logs_hpds[phs].keys()})

series.index = pd.MultiIndex.from_tuples(series.index)
series.reset_index()


print(log_hpds[0:2])
