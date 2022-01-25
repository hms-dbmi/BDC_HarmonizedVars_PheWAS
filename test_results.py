import yaml
import os 
import json

import pandas as pd

with open("env_variables/parameters_exp.yaml", "r") as f:
    parameters_exp = yaml.load(f, Loader=yaml.SafeLoader)

date_results = "010101_000000"
phs = "phs000007"
batch_group = "1"

path_results = os.path.join(parameters_exp["results_path"], date_results)


path_association = os.path.join(path_results, "association_statistics", phs, batch_group + ".zip")
path_descriptive = os.path.join(path_results, "descriptive_statistics", phs, batch_group + ".zip")

path_logs_association_statistics = os.path.join(path_results, "logs_association_statistics", phs, batch_group + ".zip")
path_logs_query = os.path.join(path_results, "logs_hpds_query", phs, batch_group + ".json")
path_monitor = os.path.join(path_results, "monitor_process.tsv")


association_table = pd.read_csv(path_association)
descriptive_table = pd.read_csv(path_descriptive)
logs_association_table = pd.read_csv(path_logs_association_statistics)
with open(path_logs_query, "r") as json_file:
    logs_query = json.load(json_file)
monitoring_table = pd.read_table(path_monitor, header=None)


iteration_time = pd.to_datetime(monitoring_table.iloc[:, 3], 
              format="%y/%m/%d_%H:%M:%S") -\
pd.to_datetime(monitoring_table.iloc[:, 2], 
              format="%y/%m/%d_%H:%M:%S")
iteration_time
descriptive_table.info()
descriptive_table.head()
association_table.info()
association_table.head()
logs_association_table.info()
logs_association_table.head()
monitoring_table.info()
monitoring_table.head()    
