import yaml
import os 
import json

import pandas as pd

with open("env_variables/parameters_exp.yaml", "r") as f:
    parameters_exp = yaml.load(f, Loader=yaml.SafeLoader)

date_results = "211029_024422"
phs = "phs000007"
batch_group = "1"

path_results = os.path.join(parameters_exp["results_path"], date_results)

path_association = os.path.join(path_results, "association_statistics", phs, batch_group + ".csv")
path_descriptive = os.path.join(path_results, "descriptive_statistics", phs, batch_group + ".csv")

path_logs_association_statistics = os.path.join(path_results, "logs_association_statistics", phs, batch_group + ".csv")
path_logs_query = os.path.join(path_results, "logs_hpds_query", phs, batch_group + ".json")
path_monitor = os.path.join(path_results, "monitor_process.csv")

association_table = pd.read_csv(path_association, low_memory=False)
descriptive_table = pd.read_csv(path_descriptive, low_memory=False)
logs_association_table = pd.read_csv(path_logs_association_statistics)
with open(path_logs_query, "r") as json_file:
    logs_query = json.load(json_file)
monitoring_table = pd.read_csv(path_monitor, header=None, index_col = None)
monitoring_table = monitoring_table.set_axis(monitoring_table_colnames, axis=1)

monitoring_table_colnames = [
    "phs",
    "batch_group",
    "nb_independent_vars", 
    "nb_filtered_independent_vars", 
    "nb_dependent_vars", 
    "time_start", 
    "time_stop",
    "duration"
]

iteration_time = monitoring_table.iloc[:, 7]
iteration_time
descriptive_table.info()
descriptive_table.head()
association_table.info()
association_table.head()
logs_association_table.info()
logs_association_table.head()
monitoring_table.info()
monitoring_table.head()    

# Monitoring_table
monitoring_table.sort_values(["phs", "batch_group"], ascending=True)
