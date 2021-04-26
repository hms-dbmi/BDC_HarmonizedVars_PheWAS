from typing import List, Tuple

import pandas as pd
import numpy as np 


def _query_HPDS(variables: list,
                resource,
                qtype="select") -> pd.DataFrame:
    query = resource.query()
    if qtype == "select":
        query.select().add(variables)
    else:
        raise ValueError("only select implemented right now")
    return query.getResultsDataFrame(timeout=60)


def _independent_var_selection(subset_variablesDict,
                               phenotypes=True,
                               nb_categories: tuple=None, 
                               ):
    if phenotypes is True:
        mask_pheno = subset_variablesDict["HpdsDataType"] == "phenotypes"
        subset_variablesDict = subset_variablesDict.loc[mask_pheno,:]
    if nb_categories is not None:
        mask_modalities = subset_variablesDict["categorical"] == False | subset_variablesDict["nb_modalities"]\
            .between(*nb_categories, inclusive=True)
        subset_variablesDict = subset_variablesDict.loc[mask_modalities, :]

    return subset_variablesDict


def _LRT(dependent_var_name: str,
        independent_var_names: List[str],
        study_df: pd.DataFrame) -> Tuple[dict, dict]:
    
    from statsmodels.discrete.discrete_model import Logit
    from scipy.linalg import LinAlgError
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    from tqdm import tqdm
    
    dic_pvalues = {}
    dic_errors = {}
    for independent_var_name in tqdm(independent_var_names, position=0, leave=True):
        print(independent_var_name)
        subset_df = study_df.loc[:, [dependent_var_name, independent_var_name]]\
                            .dropna(how="any")\
                            .copy()
        
        if subset_df.shape[0] == 0:
            dic_pvalues[independent_var_name] = np.NaN
            dic_errors[independent_var_name] = "All NaN"
            continue
        
        if subset_df[independent_var_name].dtype in ["object", "bool"]:
            subset_df = pd.get_dummies(subset_df,
                                       columns=[independent_var_name],
                                       drop_first=False)\
                          .iloc[:, 0:-1]
        y = subset_df[dependent_var_name].cat.codes
        X = subset_df.drop(dependent_var_name, axis=1)\
                                .assign(intercept = 1)
        model = Logit(y, X)
        try:
            results = model.fit(disp=0)
            params = results.params.drop("intercept", axis=0)
            conf = np.exp(results.conf_int().drop("intercept", axis=0))
            conf['OR'] = np.exp(params)
            conf["pvalue"] = results.pvalues.drop("intercept", axis=0)
            conf = conf.rename({0: 'lb', 1: 'ub'}, axis=1)
            dic_or = conf.to_dict(orient="index")
            dic_pvalues[independent_var_name] = {"llr_pvalue": results.llr_pvalue, **dic_or}
        except (LinAlgError, PerfectSeparationError) as e:
            dic_pvalues[independent_var_name] = np.NaN
            dic_errors[independent_var_name] = str(e)
    return dic_pvalues, dic_errors


def PheWAS(study_df: pd.DataFrame,
           dependent_var_name: str) -> Tuple[dict, dict]:
    print("Shape of retrieved HPDS dataframe {0}".format(study_df.shape))
    study_df[dependent_var_name] = study_df[dependent_var_name].astype("category")
    independent_var_names = list(set(study_df.columns.tolist()) - {dependent_var_name})
    return _LRT(dependent_var_name, independent_var_names, study_df)


def manhattan_plot(df_pvalues,
                   threshold_group_cat=5,
                   title_plot="Statistical Association Between Exposition Status and Phenotypes"):
    adjusted_alpha = 0.05 / len(df_pvalues["pvalue"])
    df_pvalues["p_adj"] = df_pvalues["pvalue"] / len(df_pvalues["pvalue"])
    df_pvalues['log_p'] = -np.log10(df_pvalues['pvalue'])
    
    df_pvalues["group"] = df_pvalues.index
    group_counts = df_pvalues["group"].value_counts()
    group_to_merge = group_counts[group_counts < threshold_group_cat].index
    mask_group_to_merge = df_pvalues["group"].isin(group_to_merge)
    df_pvalues.loc[mask_group_to_merge, "group"] = "Other"
    df_pvalues = df_pvalues.sort_values(by="group", axis=0)
    
    dic_renaming = {
        'Genetic Epidemiology of COPD (COPDGene)': 'COPDGene',
        'Genetic Epidemiology Network of Arteriopathy (GENOA)': 'GENOA',
        'NHLBI TOPMed: Genetics of Cardiometabolic Health in the Amish': 'Genetics',
        'Genome-wide Association Study of Adiposity in Samoans': 'GEWAS Samoans',
        'Genetics of Lipid Lowering Drugs and Diet Network (GOLDN) Lipidomics Study': 'GOLDN',
        'Heart and Vascular Health Study (HVH)': 'HVH'
    }
    df_pvalues["group"] = df_pvalues["group"].replace(dic_renaming)
    
    df_pvalues["variable"] = df_pvalues["variable"].str.replace("[0-9]+[A-z]*", "").to_frame()
    order_studies = df_pvalues.index.get_level_values(0).unique().tolist()[::-1]
    # df_pvalues = df_pvalues.reindex(order_studies, level=0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = plt.get_cmap('Set1')
    x_labels = []
    x_labels_pos = []
    
    y_lims = (0, df_pvalues["log_p"].max(skipna=True) + 50)
    threshold_top_values = df_pvalues["log_p"].sort_values(ascending=False)[0:6].iloc[-1]
    
    df_pvalues["ind"] = np.arange(1, len(df_pvalues) + 1)
    # df_pvalues["group"] = df_pvalues["group"].str.replace("[0-9]", "")
    df_grouped = df_pvalues.groupby(('group'))
    for num, (name, group) in enumerate(df_grouped):
        group.plot(kind='scatter', x='ind', y='log_p', color=colors.colors[num % len(colors.colors)], ax=ax, s=20)
        x_labels.append(name)
        x_labels_pos.append(
            (group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0]) / 2))  # Set label in the middle
        
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
    return

