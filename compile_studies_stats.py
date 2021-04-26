import re
import os

import pandas as pd


def gather_csvs():
    dir_result_studies = "studies_stats/by_phs"
    list_csv_studies = [os.path.join(dir_result_studies, f) for f \
                        in os.listdir(dir_result_studies) if re.search("\.csv$", f)]
    list_csv_studies.append("./studies_stats/harmonized/harmonized_stats.csv")
    list_pandas = [pd.read_csv(csv, index_col=0, header=[0, 1]) for csv in list_csv_studies]
    return pd.concat(list_pandas)


studies_info = pd.read_csv("env_variables/studies_info.csv").set_index("BDC_study_name")
studies_stats = gather_csvs()\
    .join(studies_info["harmonized"], how="left")\
    .loc[lambda df: (df["harmonized"] == True) | (df.index == "DCC Harmonized data set"), :]\
    .drop("harmonized", axis=1)
studies_stats.columns = pd.MultiIndex.from_tuples(studies_stats.columns.tolist())
at = ("Number variables with non-null values", "Mean non-null value count per variable")
studies_stats[at] = studies_stats[at].round(1)
at = ("Number variables with non-null values", "Median non-null value count per variable")
studies_stats[at] = studies_stats[at].astype(int)

studies_stats = studies_stats.rename(columns={"Total number subjects": "Population Count"})
studies_stats.to_csv("studies_stats/studies_stats.csv")

###
