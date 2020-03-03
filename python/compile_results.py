import re

import pandas as pd


import os

with open("phs_list.txt") as f:
    list_phs = f.read().splitlines()
dir_results_csv = "./studies_stats/"
list_csv = [file_name for file_name in os.listdir(dir_results_csv) if re.search("\.csv$", file_name)]


def test_complete_results(list_phs, list_csv):
    missing_csv = [phs for phs in list_phs if not any(re.search(phs, csv) for csv in list_csv)]
    if len(missing_csv) != 0:
        print("missing csv {0}".format(missing_csv))
        return missing_csv
    else:
        print("No missing csv")
        return 
        
def gather_csvs(list_csv):
    list_pandas = [pd.read_csv(os.path.join("studies_stats", csv), index_col=0, header=[0, 1]) for csv in list_csv]
    list_pandas.append(pd.read_csv("studies_stats/harmonized_stats.csv", index_col=0, header=[0, 1]))
    return pd.concat(list_pandas)


studies_stats = gather_csvs(list_csv)    
studies_stats.to_csv("studies_stats/studies_stats.csv")
