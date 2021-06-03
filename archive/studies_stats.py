import os
from pprint import pprint 
from argparse import ArgumentParser
from ast import literal_eval
import json

import pandas as pd

from env_variables.env_variables import TOKEN, RESOURCE_ID, PICSURE_NETWORK_URL, UPSTREAM
from python_lib.querying_hpds import get_HPDS_connection,\
    get_one_study,\
    get_whole_dic, \
    query_runner
from python_lib.utils import get_multiIndex_variablesDict
from python_lib.quality_checking import quality_filtering, get_study_info

parser = ArgumentParser()
parser.add_argument("--phs", dest="phs", type=str, default=None)
parser.add_argument("--batch_group", dest="batch_group", type=int, default=None)
parser.add_argument("--upstream", dest="upstream", type=str, default="False")
args = parser.parse_args()
phs = args.phs
batch_group = args.batch_group
upstream = UPSTREAM if args.upstream() == "False" else True


resource = get_HPDS_connection(TOKEN, PICSURE_NETWORK_URL, RESOURCE_ID)

if upstream:
    variablesDict = get_whole_dic(resource, batch_size=5000, write=False)
    variablesDict = get_multiIndex_variablesDict(variablesDict)
    variablesDict.to_csv("env_variables/multiIndex_variablesDict.csv")
else:
    variablesDict = pd.read_csv("env_variables/multiIndex_variablesDict.csv",
                                low_memory=False)

    
if phs is not None:
    print("entering phs: {}".format(phs))
    studies_info = pd.read_csv("env_variables/studies_info.csv",
                               index_col=0,
                               converters={"phs_list": literal_eval})
    study_name = studies_info.loc[phs, "BDC_study_name"]
    original_df = get_one_study(resource,
                                phs,
                                studies_info,
                                variablesDict,
                                harmonized_vars=True,
                                low_memory=False)
    print("original df shape {0}".format(original_df.shape))
elif batch_group is not None:
    print("entering batch_group: {}".format(batch_group))
    variables_to_select = variablesDict.loc[variablesDict["batch_group"] == batch_group, "name"].tolist()
    original_df = query_runner(resource,
                               to_select=variables_to_select,
                               low_memory=False)
    print("original df shape {0}".format(original_df.shape))
    original_df.to_csv("env_variables/variables/" + str(batch_group) + ".csv", index=False)

else:
    raise ValueError("--batch_group or --phs argument should be provided")

print("shape original_df: {0}".format(original_df.shape))
filtered_df = quality_filtering(original_df)
print("shape filtered_df: {0}".format(filtered_df.shape))

# variables_info = get_study_info(original_df, filtered_df)
# pprint(variables_info)
#
# index_df = study_name if phs is not None else batch_group
# var_info_df = pd.DataFrame({(i, j): variables_info[i][j]
#                             for i in variables_info.keys()
#                             for j in variables_info[i].keys()},
#                             index=[index_df]
#                             )
# variables_dic = {
#     "non phenotypic variables": list(set(original_df.columns) - set(filtered_df.columns)),
#     "phenotypic variables": list(filtered_df.columns)
# }
#
# path_results = os.path.join("results/studies_stats", str(phs), batch_group)
# if not os.path.isdir(path_results):
#     os.mkdirs(path_results)
#
# var_info_df.to_csv(os.path.join(path_results, "stats.csv"))
# with open(os.path.join(path_results, "_vars.json"), "w+") as j:
#     json.dump(variables_dic, j)

# if phs is not None:
# elif batch_group is not None:
#     var_info_df.to_csv(os.path.join("../studies_stats/batch_group", str(batch_group) + "_stats.csv"))
#     with open(os.path.join("../studies_stats/batch_group", str(batch_group) + "_vars.json"), "w+") as j:
#         json.dump(variables_dic, j)
# else:
#     raise ValueError("--batch_group or --phs argument should be provided")

print("went through!")

