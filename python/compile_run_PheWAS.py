import json
import os 
import re 

import pandas as pd



list_files = [f for f in os.listdir("results/pvalues") if re.search("_pvalues.json", f)]

studies_info = pd.read_csv("studies_info.csv").loc[:, ["phs", "BDC_study_name"]]\
.set_index("phs").iloc[:, 0]
#studies_info["file_name"] = studies_info["phs"] + "_pvalues.json"
#dic_file_study_name = {file_name: study_name for file_name, study_name in studies_info.set_index("file_name")["BDC_study_name"].iteritems()}

dic_pvalues = {}
#for file_name in list_files:
with open("sublist_phs.txt", "r") as f:
    phs_list = f.read().splitlines()
for phs in phs_list:
    #with open(os.path.join("results/pvalues", file_name), "r") as f:
    #    dic_pvalues[dic_file_study_name[phs]] = json.load(f)
    with open(os.path.join("results/pvalues", phs + "_pvalues.json"), "r") as f:
        dic_pvalues[studies_info[phs]] = json.load(f)

#df_pvalues = pd.DataFrame.from_dict({(file_name, variable): pvalue for 
#                                    file_name, dic_var_pvalues in dic_pvalues.items() for
#                                    variable, pvalue in dic_var_pvalues.items()
#}, orient="index")

### Changes because some scenario didn't run
dic_pvalues.pop("NHLBI TOPMed: Genetics of Cardiometabolic Health in the Amish")
dic_pvalues.pop("Heart and Vascular Health Study (HVH)")
df_pvalues = pd.DataFrame.from_dict({(study_name, dependent_var_name, variable): pvalue for 
                                     study_name, dependent_var_dic in dic_pvalues.items() for
                                    dependent_var_name, dic_var_pvalues in dependent_var_dic.items() for
                                    variable, pvalue in dic_var_pvalues.items()
}, orient="index")

df_pvalues.index = pd.MultiIndex.from_tuples(df_pvalues.index)
df_pvalues = df_pvalues.reset_index(-1, drop=False)
df_pvalues.columns = ["variable", "pvalue"]
df_pvalues = df_pvalues.dropna(subset=["pvalue"])
mask_0 = df_pvalues["pvalue"] == 0
df_pvalues = df_pvalues.loc[~mask_0,:]
df_pvalues.index.value_counts().to_frame()

multiIndex_variablesDict = pd.read_csv("multiIndex_variablesDict.csv", index_col=list(range(0, 13)), low_memory=False)
simplified_varnames = multiIndex_variablesDict.loc[:, ["varName", "simplified_varName"]].reset_index(drop=True)

df_pvalues = df_pvalues.join(simplified_varnames.rename({"varName": "variable"}, axis=1).set_index("variable"), on= "variable", how="left")
df_pvalues.to_csv("df_pvalues_bis.csv")

####################