import json
import os
import re

import pandas as pd
import numpy as np

from python_lib.errors import ExtensionError
path_exp = "./results_server/results/210612_004737/"


class CompilePheWAS_Results():
    
    def __init__(self,
                 path_exp):
        self.path_logs_stats = os.path.join(path_exp, "logs_association_statistics")
        self.path_logs_hpds = os.path.join(path_exp, "logs_association_statistics")
        self.path_association_statistics_results = os.path.join(path_exp, "association_statistics")

    @staticmethod
    def download_dropbox(function):
        def _download_dropbox(*args, **kwargs):
            if args[0].dropbox is True:
                # TODO: to be implemented eventually
                pass
            return function(*args, **kwargs)
    
        return _download_dropbox

    @staticmethod
    @download_dropbox
    def read_file(file_name, dir_path, *args, **kwargs):
        file_path = os.path.join(dir_path, file_name)
        extension_regex = re.compile(r"\.[a-z]+$")
        extension = re.search(extension_regex, file_path).group()
        if extension == ".csv":
            output = pd.read_csv(file_path, *args, **kwargs)
        elif extension == ".json":
            with open(file_path, "w+") as json_file:
                output = json.load(json_file, *args, **kwargs)
        elif extension == ".txt":
            with open(file_path, "w+") as text_file:
                output = text_file.read()
        else:
            raise ExtensionError
        return output

    def get_logs_hpds(self):
        pass
        return
    
    def get_logs_quality_checking(self):
        pass
        return
    
    def get_logs_statistics(self):
        pass
        return
    
    def get_descriptive_statistics(self):
        pass
        return
    
    def compile_association_statistics(self):
        # pd.read_csv(self.path_association_statistics_results)
        return
        

if __name__ == '__main__':
    path_exp = "./results_server/results/210612_004737/"
    path_logs_stats = os.path.join(path_exp, "logs_association_statistics")
    path_logs_hpds = os.path.join(path_exp, "logs_association_statistics")
    path_association_statistics_results = os.path.join(path_exp, "association_statistics")
    path_test_df = os.path.join(path_association_statistics_results, "phs000007/0.csv")
    path_test_df = os.path.join(path_association_statistics_results, "phs000007/23.csv")
    test_df = pd.read_csv(path_test_df)
    test_df.to_pickle(os.path.join(path_association_statistics_results, "phs000007/0.zip"),
                      compression="infer")
        
    test_df.to_pickle(os.path.join(path_association_statistics_results, "phs000007/0.gzip"),
                      compression="infer")
    
    test_df.to_pickle(os.path.join(path_association_statistics_results, "phs000007/23.zip"),
                      compression="infer")
    test_df.to_pickle(os.path.join(path_association_statistics_results, "phs000007/23.pickle"),
                      compression="infer")
    test2 = pd.read_pickle(os.path.join(path_association_statistics_results, "phs000007/23.zip"))
    test2 = pd.read_pickle(os.path.join(path_association_statistics_results, "phs000007/0.zip"))
    path_test_df = os.path.join(path_association_statistics_results, "phs000007/0.csv")
        
        
    

## Get the number of phenotypic variables
study_info = pd.read_csv("env_variables/studies_info_manual_dont_erase.csv", index_col=0)
df_eligible_variables = pd.read_csv("env_variables/list_eligible_variables.csv")\
    .join(study_info["BDC_study_name"], on="phs")
df_eligible_variables.BDC_study_name.unique().shape[0]
df_eligible_variables_value_counts = df_eligible_variables["BDC_study_name"]\
    .value_counts()
sum_df_eligible_variables_counts = df_eligible_variables_value_counts.sum()
df_eligible_variables_value_counts\
    .append(pd.Series([sum_df_eligible_variables_counts], index=["Total"]), ignore_index=False)\
    .rename_axis("Name Study")\
    .rename("Phenotypic Variables Count")\
    .to_frame()\
    .to_csv("exports_presentation/tables/ind_variable_counts_per_studies.csv")



def read_json_associations(path_dir):
    import glob, os
    from pathlib import Path

    list_logs = []
    for phs in os.listdir(path_dir):
        print(phs)
        path_subdir = os.path.join(path_dir, phs)
        print(path_subdir)
        for batch_group in os.listdir(path_subdir):
            print(batch_group)
            path_file = os.path.join(path_subdir,  batch_group)
            print(path_file)
            with open(path_file, "r") as json_file:
                logs = json.load(json_file)
                logs = {}
            list_logs.append(logs)
    df_logs = pd.concat([pd.DataFrame.from_dict(df) for df in list_logs])
    return df_logs
    

def read_association_results(path_dir):
    
    list_results = []
    for phs in os.listdir(path_dir):
        print(phs)
        path_subdir = os.path.join(path_dir, phs)
        print(path_subdir)
        for batch_group in os.listdir(path_subdir):
            print(batch_group)
            path_file = os.path.join(path_subdir,  batch_group)
            print(path_file)
            results = pd.read_csv(path_file)
            pvalues = results.loc[lambda df: df["indicator"].str.contains("LRT") == True, ["value", "independent_var_id", "dependent_var_id"]]
            list_results.append(pvalues)
    
    df_results_pvalues = pd.concat([pd.DataFrame.from_dict(df) for df in list_results])
    
    return df_results_pvalues


    
pvalues2 = read_association_results(path_dir)

# Number of eligible variables after quality checking eligible variables
quality_checking = pd.read_csv("quality_checking")
# Information to gather
## Number of batch runs
monitor_process = pd.read_table(os.path.join(path_exp, "monitor_process.tsv"),
                                sep="\t",
                                header=None)\
    .rename({0: "phs", 1:"batch_group"}, axis=1)\
    .set_index(["phs", "batch_group"])

df_eligible_variables.set_index(["phs", "batch_group"])\
    .join(monitor_process, how="inner")

## Number of studies



## Number of pvalues
## Number of associations ran
df_results_pvalues = read_association_results(os.path.join(path_exp, "association_statistics"))
df_results_pvalues.shape[0]
# Bonferonni threshold for significance
bonferonni = 0.05/df_results_pvalues.shape[0]

## Number of failure
df_results_pvalues.value.isna().value_counts()
## Number of results below threshold
df_results_pvalues.loc[lambda df: df["value"]< bonferonni, :]
df_results_pvalues.loc[lambda df: df["value"]< 0.05, :]
## Number of counts


## Parameters to tweak

## Size of results


## Expected size of results

## Some

## Technical challenges
--> adapt the instance
## Methodological challenges
--> how to produce meaningful results?
    --> Cross with PubMed results
    --> Produce result explorator
- What I suspect is that there is some categorical variables that create a lot of subcategories, creating a lot of supplemental models to be ran

results.info()
df_results = pd.read_csv(path_dir + "")
    
    
    for sub_dir in sub_directories:
        os.chdir(sub_dir)
        for batch_group in glob.glob("*.json"):
            all_json_files.append(sub_dir + "/" + batch_group)

    # Get back to original working directory
    os.chdir(working_directory)
    
    list_of_dfs = [pd.read_json(x) for x in all_json_files]
    
    return jsons
path_association_results = "results/archives/association_statistics/phs000007/0.csv"
association_statistics = pd.read_csv(path_association_results)
association_statistics.info()
len(association_statistics.independent_var_name.unique())
len(association_statistics.dependent_var_name.unique())
test = pd.pivot(association_statistics,
         columns="indicator",
         values="value")
test.columns
test.coeff_LogR.isna().value_counts()
# from datetime import datetime, timezone
# dt = datetime(2020, 6, 1)
# timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
# print(timestamp)
root_dir = "results/associations"
list_files = [f for f in os.listdir(root_dir, ) if re.match(".*_pvalues.json$", f)]
studies_info = pd.read_csv("env_variables/studies_info.csv").loc[:, ["phs", "BDC_study_name"]]\
                   .set_index("phs").iloc[:, 0]
#studies_info["file_name"] = studies_info["phs"] + "_pvalues.json"
#dic_file_study_name = {file_name: study_name for file_name, study_name in studies_info.set_index("file_name")["BDC_study_name"].iteritems()}

dic_pvalues = {}
#for file_name in list_files:
with open("env_variables/phs_list.txt", "r") as f:
    phs_list = f.read().splitlines()

for file_path in list_files:
    phs = re.search("phs[0-9]+(?=_)", file_path).group()
    with open(os.path.join("results/associations", file_path), "r") as f:
        dic_pvalues[phs] = json.load(f)


#df_pvalues = pd.DataFrame.from_dict({(file_name, variable): pvalue for
#                                    file_name, dic_var_pvalues in dic_pvalues.items() for
#                                    variable, pvalue in dic_var_pvalues.items()
#}, orient="index")

### Changes because some scenario didn't run
# dic_pvalues.pop("NHLBI TOPMed: Genetics of Cardiometabolic Health in the Amish")
# dic_pvalues.pop("Heart and Vascular Health Study (HVH)")
# df_pvalues = pd.DataFrame.from_dict({(study_name, dependent_var_name, variable): pvalue["llr_pvalue"] for
#                                      study_name, dependent_var_dic in dic_pvalues.items() for
#                                     dependent_var_name, dic_var_pvalues in dependent_var_dic.items() for
#                                     variable, pvalue in dic_var_pvalues.items()
# }, orient="index")
flat_dic = {}
flat_dic_subvar = {}
for study_name, dependent_var_dic in dic_pvalues.items():
    for dependent_var_name, dic_var_pvalues in dependent_var_dic.items():
        for variable, pvalue in dic_var_pvalues.items():
            if isinstance(pvalue, float):
                flat_dic[(study_name, dependent_var_name, variable)] = pvalue
            elif isinstance(pvalue, dict):
                flat_dic[(study_name, dependent_var_name, variable)] = pvalue.pop("llr_pvalue")
                for subvar, params in pvalue.items():
                    for param, param_value in params.items():
                        flat_dic_subvar[(study_name, dependent_var_name, variable, subvar, param)] = param_value
            else:
                raise TypeError("error for {}".format((study_name, dependent_var_name, variable)))

df_pvalues = pd.Series(flat_dic, name="pvalues").reset_index()
df_params = pd.Series(flat_dic_subvar, name="param").reset_index()
df_pvalues.to_csv("results/df_results/df_pvalues.csv")
df_params.to_csv("results/df_results/df_params.csv")

import matplotlib.pyplot as plt

# df_pvalues.set_index("level_0")["pvalues"].plot(kind="hist", bins=30)

# df_pvalues.index = pd.MultiIndex.from_tuples(df_pvalues.index)
# df_pvalues = df_pvalues.reset_index(-1, drop=False)
# df_pvalues.columns = ["variable", "pvalue"]
# df_pvalues = df_pvalues.dropna(subset=["pvalue"])
# mask_0 = df_pvalues["pvalue"] == 0
# df_pvalues = df_pvalues.loc[~mask_0,:]
# df_pvalues.index.value_counts().to_frame()
#
# multiIndex_variablesDict = pd.read_csv("multiIndex_variablesDict.csv", index_col=list(range(0, 13)), low_memory=False)
# simplified_varnames = multiIndex_variablesDict.loc[:, ["varName", "simplified_varName"]].reset_index(drop=True)
#
# df_pvalues = df_pvalues.join(simplified_varnames.rename({"varName": "variable"}, axis=1).set_index("variable"), on= "variable", how="left")
# df_pvalues.to_csv("df_pvalues_bis.csv")

####################


# group_counts = df_pvalues["group"].value_counts()
# group_to_merge = group_counts[group_counts < threshold_group_cat].index
# mask_group_to_merge = df_pvalues["group"].isin(group_to_merge)
# df_pvalues.loc[mask_group_to_merge, "group"] = "Other"
# df_pvalues = df_pvalues.sort_values(by="group", axis=0)

# dic_renaming = {
#     'Genetic Epidemiology of COPD (COPDGene)': 'COPDGene',
#     'Genetic Epidemiology Network of Arteriopathy (GENOA)': 'GENOA',
#     'NHLBI TOPMed: Genetics of Cardiometabolic Health in the Amish': 'Genetics',
#     'Genome-wide Association Study of Adiposity in Samoans': 'GEWAS Samoans',
#     'Genetics of Lipid Lowering Drugs and Diet Network (GOLDN) Lipidomics Study': 'GOLDN',
#     'Heart and Vascular Health Study (HVH)': 'HVH'
# }
# df_pvalues["group"] = df_pvalues["group"].replace(dic_renaming)

# df_pvalues["variable"] = df_pvalues["variable"].str.replace("[0-9]+[A-z]*", "").to_frame()
# order_studies = df_pvalues.index.get_level_values(0).unique().tolist()[::-1]
# df_pvalues = df_pvalues.reindex(order_studies, level=0)
    
    pair_ind = 0  # To shift label which might overlap because to close
    for n, row in group.iterrows():
        #        if pair_ind %2 == 0:
        #            shift = 1.1
        #        else:
        #            shift = -1.1
        if row["log_p"] > threshold_top_values:
            ax.text(row['ind'] + 3, row["log_p"] + 0.05, row["simplified_varName"], rotation=0, alpha=1, size=8,
                    color="black")
#            pair_ind += 1

ax.set_xticks(x_labels_pos)
ax.set_xticklabels(x_labels)
ax.set_xlim([0, len(df_pvalues) + 1])
ax.set_ylim(y_lims)
ax.set_ylabel('-log(p-values)', style="italic")
ax.set_xlabel('Phenotypes', fontsize=15)
ax.axhline(y=-np.log10(adjusted_alpha), linestyle=":", color="black", label="Bonferonni Adjusted Threshold")
plt.xticks(fontsize=9, rotation=30)
plt.yticks(fontsize=8)
plt.title(title_plot,
          loc="left",
          style="oblique",
          fontsize=20,
          y=1)
xticks = ax.xaxis.get_major_ticks()
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, loc="upper left")
plt.show()