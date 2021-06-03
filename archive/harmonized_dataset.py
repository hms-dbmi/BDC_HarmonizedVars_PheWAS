import json 

import pandas as pd

from env_variables.env_variables import TOKEN, RESOURCE_ID, PICSURE_NETWORK_URL
from python_lib.querying_hpds import get_HPDS_connection, query_runner
from python_lib.quality_checking import quality_filtering, get_study_info


resource = get_HPDS_connection(TOKEN, PICSURE_NETWORK_URL, RESOURCE_ID)

dic_harmonized = resource.dictionary().find("Harmonized").DataFrame()
harmonized_variables = dic_harmonized.index.tolist()
harmonized_df = query_runner(resource=resource, 
                             to_anyof=harmonized_variables,
                             low_memory=False)

#harmonized_df = harmonized_df.set_index("Patient ID")
quality_harmonized = quality_filtering(harmonized_df)
study_info = get_study_info(harmonized_df, quality_harmonized)
study_info_df = pd.DataFrame({(i, j): study_info[i][j]
                              for i in study_info.keys()
                              for j in study_info[i].keys()},
                             index=["DCC Harmonized data set"]
                             )
study_info_df.to_csv("results/studies_stats/harmonized/harmonized_stats.csv")


# Statistics about null values in harmonized dataset
def get_stat_harmonized_variables():
    non_null = quality_harmonized.notnull().sum()
    non_null.name = "non-null values"
    unique = quality_harmonized.nunique()
    unique.name = "unique values"

    renaming_harmonized_df = pd.read_csv("env_variables/renaming_harmonized_variables_manual.csv") \
        .set_index("harmonized_complete_name")

    variables_stats = non_null.to_frame().join(unique, how="left")\
        .rename_axis("harmonized variables")\
        .join(renaming_harmonized_df)
    variables_stats.to_csv("studies_stats/harmonized_variables_stats.csv")
    return
get_stat_harmonized_variables()


variables_dic = {
    "ID variables": list(set(harmonized_df.columns) - set(quality_harmonized.columns)),
    "phenotypic variables": quality_harmonized.columns[~quality_harmonized.columns.str.contains("age at measurement")].tolist(),
    "age variables": quality_harmonized.columns[quality_harmonized.columns.str.contains("age at measurement")].tolist()
}
with open("results/list_variables/harmonized_vars_list.json", "w+") as j:
    json.dump(variables_dic, j)

