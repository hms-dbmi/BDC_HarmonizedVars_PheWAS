import json 

import pandas as pd

from python_lib.wrappers import get_HPDS_connection, query_runner
from python_lib.descriptive_scripts import quality_filtering, get_study_variables_info

PICSURE_network_URL = "https://picsure.biodatacatalyst.nhlbi.nih.gov/picsure"
resource_id = "02e23f52-f354-4e8b-992c-d37c8b9ba140"
token_file = "token.txt"

with open("token.txt", "r") as f:
    token = f.read()

resource = get_HPDS_connection(token, PICSURE_network_URL, resource_id)

dic_harmonized = resource.dictionary().find("Harmonized").DataFrame()
harmonized_variables = dic_harmonized.index.tolist()

harmonized_df = query_runner(resource=resource, 
                             to_anyof=harmonized_variables,
                            low_memory=False)

#harmonized_df = harmonized_df.set_index("Patient ID")
quality_harmonized = quality_filtering(harmonized_df)
variables_info = get_study_variables_info(harmonized_df, quality_harmonized)


var_info_df = pd.DataFrame({(i, j): variables_info[i][j]
                            for i in variables_info.keys()
                            for j in variables_info[i].keys()},
                       index=["DCC Harmonized data set"]
                            )

var_info_df.to_csv("studies_stats/harmonized_stats.csv")


non_null = quality_harmonized.notnull().sum()
non_null.name = "non-null values"
unique = quality_harmonized.nunique()
unique.name = "unique values"

detailed_stats = non_null.to_frame().join(unique, how="left")\
    .rename_axis("harmonized variables")
detailed_stats.to_csv("studies_stats/harmonized_details_stats.csv")


variables_dic = {
    "ID variables": list(set(harmonized_df.columns) - set(quality_harmonized.columns)),
    "phenotypic variables": quality_harmonized.columns[~quality_harmonized.columns.str.contains("age at measurement")].tolist(),
    "age variables": quality_harmonized.columns[quality_harmonized.columns.str.contains("age at measurement")].tolist()
}
with open("studies_stats/harmonized_vars.json", "w+") as j:
    json.dump(variables_dic, j)

