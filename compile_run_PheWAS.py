import json
import os
import re

import pandas as pd
import numpy as np

path_logs = "results/archives/logs_association_statistics/phs000007/0.pickle"
with open(path_logs, "r") as f:
    logs = json.load(f)

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