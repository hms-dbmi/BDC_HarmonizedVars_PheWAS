import os
from pprint import pprint 
from argparse import ArgumentParser
from ast import literal_eval
import json

import pandas as pd

from querying_hpds import get_HPDS_connection, get_one_study, get_whole_dic
from utils import get_multiIndex_variablesDict
from descriptive_scripts import quality_filtering, get_study_variables_info

parser = ArgumentParser()
parser.add_argument("--phs", dest="phs", type=str, default=None)
parser.add_argument("--batch_group", dest="batch_group", type=int, default=None)
parser.add_argument("--upstream", dest="upstream", type=str, default="False")
args = parser.parse_args()
phs = args.phs
batch_group = args.batch_group
upstream = False if args.upstream() == "False" else True

PICSURE_network_URL = "https://picsure.biodatacatalyst.nhlbi.nih.gov/picsure"
resource_id = "02e23f52-f354-4e8b-992c-d37c8b9ba140"
token_file = "token.txt"

with open("token.txt", "r") as f:
    token = f.read()

resource = get_HPDS_connection(token, PICSURE_network_URL, resource_id)

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
    original_df = get_one_study(resource, phs, studies_info, variablesDict, harmonized_vars=True, low_memory=False)
    print("original df shape {0}".format(original_df.shape))
elif batch_group is not None:
    print("entering batch_group: {}".format(batch_group))
    variables_to_select = variablesDict.loc[variablesDict["batch_group"] == batch_group, "name"].tolist()
    #original_df = query_runner(resource, 
    #                          to_select=variables_to_select,
    #                          low_memory=False)
    print("original df shape {0}".format(original_df.shape))
    original_df.to_csv("env_variables/variables/" + str(batch_group) + ".csv", index=False)

else:
    raise ValueError("--batch_group or --phs argument should be provided")

print("shape original_df: {0}".format(original_df.shape))
filtered_df = quality_filtering(original_df)
print("shape filtered_df: {0}".format(filtered_df.shape))

variables_info = get_study_variables_info(original_df, filtered_df)
pprint(variables_info)

index_df = study_name if phs is not None else batch_group
var_info_df = pd.DataFrame({(i, j): variables_info[i][j]
                            for i in variables_info.keys()
                            for j in variables_info[i].keys()},
                            index=[index_df]
                            )
variables_dic = {
    "non phenotypic variables": list(set(original_df.columns) - set(filtered_df.columns)),
    "phenotypic variables": list(filtered_df.columns)
}

if phs is not None:
    var_info_df.to_csv(os.path.join("studies_stats/by_phs", phs + "_stats.csv"))
    #original_df.to_pickle(os.path.join("env_variables/by_phs", phs + ".pickle"))
    with open(os.path.join("studies_stats/by_phs", phs + "_vars.json"), "w+") as j:
        json.dump(variables_dic, j)
elif batch_group is not None:
    var_info_df.to_csv(os.path.join("../studies_stats/batch_group", str(batch_group) + "_stats.csv"))
    with open(os.path.join("../studies_stats/batch_group", str(batch_group) + "_vars.json"), "w+") as j:
        json.dump(variables_dic, j)
else:
    raise ValueError("--batch_group or --phs argument should be provided")

print("went through!")


#def stuffs():
#    
#    !for $i in {}:
#        !python3.6 studies_stat --phs_list >> log_studies_stat.txt &
#    
#    return 
#