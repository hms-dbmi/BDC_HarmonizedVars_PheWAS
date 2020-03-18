import os
from pprint import pprint 
from argparse import ArgumentParser
from ast import literal_eval
import json

import pandas as pd

from python_lib.wrappers import get_HPDS_connection, get_one_study
from python_lib.utils import get_multiIndex_variablesDict
from python_lib.descriptive_scripts import quality_filtering, get_study_variables_info

parser = ArgumentParser()
parser.add_argument("--phs", dest="phs", type=str)

args = parser.parse_args()
phs = args.phs

PICSURE_network_URL = "https://picsure.biodatacatalyst.nhlbi.nih.gov/picsure"
resource_id = "02e23f52-f354-4e8b-992c-d37c8b9ba140"
token_file = "token.txt"

with open("token.txt", "r") as f:
    token = f.read()

resource = get_HPDS_connection(token, PICSURE_network_URL, resource_id)

studies_info = pd.read_csv("./studies_info.csv",
                           index_col=0, 
                          converters={"phs_list": literal_eval})
study_name = studies_info.loc[phs, "BDC_study_name"]
variablesDict = get_multiIndex_variablesDict(resource.dictionary().find().DataFrame())


print("getting original df")
original_df = get_one_study(resource, phs, studies_info, variablesDict, low_memory=False)
print("shape original_df: {0}".format(original_df.shape))

filtered_df = quality_filtering(original_df)
print("shape filtered_df: {0}".format(filtered_df.shape))


variables_info = get_study_variables_info(original_df, filtered_df)
pprint(variables_info)

var_info_df = pd.DataFrame({(i, j): variables_info[i][j]
                            for i in variables_info.keys()
                            for j in variables_info[i].keys()},
                       index=[study_name]
                            )

var_info_df.to_csv(os.path.join("studies_stats", phs + "_stats.csv"))

variables_dic = {
    "ID variables": list(set(original_df.columns) - set(filtered_df.columns)),
    "phenotypic variables": list(filtered_df.columns)
}
with open(os.path.join("studies_stats", phs + "_vars.json"), "w+") as j:
    json.dump(variables_dic, j)

print("went through!")


#def stuffs():
#    
#    !for $i in {}:
#        !python3.6 studies_stat --phs_list >> log_studies_stat.txt &
#    
#    return 
#
