import os
from argparse import ArgumentParser
import json

import pandas as pd

from python_lib.wrappers import get_HPDS_connection
from python_lib.PheWAS_funcs import PheWAS

parser = ArgumentParser()
parser.add_argument("--phs", dest="phs", type=str)

args = parser.parse_args()
phs = args.phs

with open("./token.txt", "r") as f:
    token = f.read()

variables_file_path = os.path.join("studies_stats", phs + "_vars.json")

with open(variables_file_path, "r") as j:
    study_variables = json.load(j)["phenotypic variables"]

print("getting resource")
resource = get_HPDS_connection(token,
                               "https://picsure.biodatacatalyst.nhlbi.nih.gov/picsure",
                               "02e23f52-f354-4e8b-992c-d37c8b9ba140")

harmonized_var_names = pd.read_csv("studies_stats/harmonized_details_stats.csv", index_col=0).index.tolist()
dependent_var_names = [harmonized_var_names[i] for i in [4, 22, 58]]

big_dic_pvalues = {} 
big_dic_errors = {}
for dependent_var_name in dependent_var_names:
    print(phs)
    print(dependent_var_name)
    print("entering PheWAS")
    dic_pvalues, dic_errors = PheWAS(study_variables, dependent_var_name, resource)
    print("PheWAS done")
    big_dic_pvalues[dependent_var_name] = dic_pvalues
    big_dic_errors[dependent_var_name] = dic_errors


with open(os.path.join("results/pvalues", phs + "_pvalues.json"), "w+") as f:
    json.dump(big_dic_pvalues, f)

with open(os.path.join("results/pvalues", phs + "_errors.json"), "w+") as f:
    json.dump(big_dic_errors, f)