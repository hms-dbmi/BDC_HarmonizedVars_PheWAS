import os
import yaml

import numpy as np
import pandas as pd

from env_variables.env_variables import TOKEN,\
    RESOURCE_ID,\
    PICSURE_NETWORK_URL,\
    batch_size
from python_lib.querying_hpds import get_HPDS_connection,\
    get_whole_dic, \
    get_studies_dictionary, \
    get_batch_groups, \
    query_runner


resource = get_HPDS_connection(TOKEN, PICSURE_NETWORK_URL, RESOURCE_ID)
studies_dictionary = get_studies_dictionary(resource)
variables_dictionary = get_whole_dic(resource, batch_size=None, write=False)

subset_variables_dictionary = variables_dictionary\
    .join(studies_dictionary, on="level_0", how="left")\
    .loc[lambda df: df["harmonized"] == True, :]

subset_variables_dictionary = subset_variables_dictionary.groupby("level_0") \
    .apply(lambda df: get_batch_groups(df, batch_size, assign=True))


subset_variables_dictionary\
    .to_csv("env_variables/multiIndex_variablesDict.csv")
subset_variables_dictionary.reset_index("level_0", drop=False)\
    .rename({"level_0": "study",
             "name": "var_name"}, axis=1)\
    .loc[:, ["study", "phs", "batch_group", "var_name"]]\
    .assign(independent_var_id=["I" + str(i) for i in range(0, subset_variables_dictionary.shape[0])])\
    .to_csv("env_variables/df_eligible_variables.csv", index=False)

subset_variables_dictionary[["phs", "batch_group"]].drop_duplicates()\
    .to_csv("env_variables/list_phs_batchgroup.csv", index=False)

# Get Harmonized variables types
renaming_harmonized_variables_manual = pd.read_csv("env_variables/renaming_harmonized_variables_manual.csv")\
    .loc[lambda df: df["renaming_variables"].notnull(), :]\
    .set_index("harmonized_complete_name")
harmonized_variables_dictionary = variables_dictionary.join(renaming_harmonized_variables_manual,
                          on="name",
                          how="inner")\
    .rename({"name": "var_name"}, axis=1)\
    .loc[:, ["var_name", "renaming_variables", "renaming_variables_nice"]]\
    .reset_index(drop=True)

list_harmonized_variables_names = harmonized_variables_dictionary["var_name"]
harmonized_variables_df = query_runner(resource, to_select=list_harmonized_variables_names)
variables_type = {}
variables_modalities = {}
for variable_name, serie in harmonized_variables_df[list_harmonized_variables_names].iteritems():
    if serie.dtype == "object":
        counts = serie.value_counts()
        variables_modalities[variable_name] = "[" + ", ".join(counts.index.tolist()) + "]"
        if counts.shape[0] >= 3:
            variables_type[variable_name] = "multicategorical"
        else:
            variables_type[variable_name] = "binary"
    else:
        variables_type[variable_name] = "continuous"
        variables_modalities[variable_name] = np.NaN
variables_type_df = pd.DataFrame.from_dict(variables_type,
                                           columns=["var_type"],
                                           orient="index")\
            .rename_axis("var_name", axis="index")
variables_modalities_df = pd.DataFrame.from_dict(variables_modalities,
                                                 columns=["modalities"],
                                                 orient="index",
                                                 dtype=str)\
    .rename_axis("var_name", axis="index")

harmonized_variables_dictionary.join(variables_type_df, on="var_name") \
    .join(variables_modalities_df, on="var_name") \
    .pipe(lambda df: df.assign(dependent_var_id=["D" + str(i) for i in range(0, df.shape[0])]))\
    .to_csv("env_variables/df_harmonized_variables.csv", index=False)

parameters = {
    "univariate": True,
    "Minimum number observations": 10,
    "threshold_crosscount": 10,
    "harmonized_variables_types": "all",
    "online": True,
    "save": False,
    "results_path": "./results",
    "storage_dropbox": False,
    "var_name_to_id_df": True
}
with open("env_variables/parameters_exp.yaml", "w+") as f:
    yaml.dump(parameters, f)
    
path_export_results = "exports_presentation"
path_tables = os.path.join(path_export_results, "tables")
path_images = os.path.join(path_export_results, "images")
if not os.path.exists(path_tables):
    os.makedirs(path_tables)
if not os.path.exists(path_images):
    os.makedirs(path_images)

