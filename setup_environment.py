import yaml

from env_variables.env_variables import TOKEN,\
    RESOURCE_ID,\
    PICSURE_NETWORK_URL,\
    batch_size
from python_lib.querying_hpds import get_HPDS_connection,\
    get_whole_dic, \
    get_studies_dictionary, \
    get_batch_groups


resource = get_HPDS_connection(TOKEN, PICSURE_NETWORK_URL, RESOURCE_ID)
studies_dictionary = get_studies_dictionary(resource)
variables_dictionary = get_whole_dic(resource, batch_size=None, write=False)

subset_variables_dictionary = variables_dictionary\
    .join(studies_dictionary, on="level_0", how="left")\
    .loc[lambda df: df["harmonized"] == True, :]

subset_variables_dictionary = subset_variables_dictionary.groupby("level_0") \
    .apply(lambda df: get_batch_groups(df, batch_size, assign=True))

subset_variables_dictionary.to_csv("env_variables/multiIndex_variablesDict.csv")
subset_variables_dictionary.reset_index("level_0", drop=False)\
    .rename({"level_0": "study"}, axis=1)\
    .loc[:, ["study", "phs", "batch_group", "name"]]\
    .to_csv("env_variables/list_eligible_variables.csv", index=False)

parameters = {
    "univariate": True,
    "Minimum number observations": 5,
    "variable_types": "categorical"
}
with open("env_variables/parameters_exp.yaml", "w+") as f:
    yaml.dump(parameters, f)
