import os
from argparse import ArgumentParser
import json

import pandas as pd

from python_lib.querying_hpds import get_HPDS_connection, query_runner
from python_lib.PheWAS_funcs import PheWAS

parser = ArgumentParser()
parser.add_argument("--phs", dest="phs", type=str, default=None)
parser.add_argument("--batch_group", dest="batch_group", type=int, default=None)
args = parser.parse_args()
phs = args.phs
batch_group = args.batch_group

with open("env_variables/token.txt", "r") as f:
    token = f.read()

if phs is not None:
    variables_file_path = os.path.join("results/studies_stats", "by_phs", phs + "_vars.json")
    with open(variables_file_path, "r") as j:
         study_variables = json.load(j)["phenotypic variables"]
elif batch_group is not None:
    variablesDict = pd.read_csv("env_variables/multiIndex_variablesDict.csv", low_memory=False)
    study_variables = variablesDict.loc[variablesDict["batch_group"] == batch_group, "name"].tolist()
    raise ValueError("phs or batch_group should be provided")
    

resource = get_HPDS_connection(token,
                               "https://picsure.biodatacatalyst.nhlbi.nih.gov/picsure",
                               "02e23f52-f354-4e8b-992c-d37c8b9ba140")

harmonized_var_df = pd.read_csv("results/studies_stats/harmonized/harmonized_details_stats.csv", index_col=0)
dependent_var_names = harmonized_var_df.index[harmonized_var_df["unique values"] == 2]

study_df = query_runner(resource=resource,
                        to_select=dependent_var_names,
                        to_anyof=study_variables,
                        result_type="DataFrame",
                        low_memory=False,
                        timeout=500)

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
