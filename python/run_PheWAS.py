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
dependent_var_name = "\\DCC Harmonized data set\\01 - Demographics\\Subject sex  as recorded by the study.\\"
print("entering PheWAS")
dic_pvalues, dic_errors = PheWAS(study_variables, dependent_var_name, resource)
print("PheWAS done")

with open(os.path.join("results/pvalues", phs + "_pvalues.json"), "w+") as f:
    json.dump(dic_pvalues, f)

with open(os.path.join("results/pvalues", phs + "_errors.json"), "w+") as f:
    json.dump(dic_errors, f)