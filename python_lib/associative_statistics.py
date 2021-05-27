from typing import List, Tuple

import pandas as pd
import numpy as np 
from statsmodels.discrete.discrete_model import MNLogit, Logit
from scipy.stats import chi2_contingency, f_oneway

from pandas.api.types import CategoricalDtype

from statsmodels.tools.sm_exceptions import PerfectSeparationError
# def _independent_var_selection(subset_variablesDict,
#                                phenotypes=True,
#                                nb_categories: tuple=None,
#                                ):
#     if phenotypes is True:
#         mask_pheno = subset_variablesDict["HpdsDataType"] == "phenotypes"
#         subset_variablesDict = subset_variablesDict.loc[mask_pheno,:]
#     if nb_categories is not None:
#         mask_modalities = subset_variablesDict["categorical"] == False | subset_variablesDict["nb_modalities"]\
#             .between(*nb_categories, inclusive=True)
#         subset_variablesDict = subset_variablesDict.loc[mask_modalities, :]
#
#     return subset_variablesDict

list_columns_names_export = [
    "dependent_var_name",
    "dependent_var_modality",
    "ref_modality_dependent",
    "dependent_var_name",
    "independent_var_modality",
    "ref_modality_independent",
    "indicator",
    "value"
]

def multicategorical_continuous(dependent_var: pd.Series,
                                independent_var: pd.Series,
                                list_columns_names_export: list):
    
    dependent_var_name = dependent_var.name
    independent_var_name = independent_var.name
    long_data = pd.DataFrame.from_dict({"dependent_var_name": dependent_var,
                                        "independent_var_name": independent_var},
                                       orient="columns")
    
    groupby_stats = long_data \
        .groupby("dependent_var_name") \
        .agg(["min", "max", "median", "mean", "count"]) \
        .droplevel(level=0, axis=1) \
        .rename({"count": "nonNA_count"}, axis=1) \
        .reset_index(drop=False) \
        .melt(id_vars="dependent_var_name",
              var_name="indicator") \
        .rename({"dependent_var_name": "dependent_var_modality"},
                axis=1)\
        .round({"value": 2})
    ###############
    
    ############# ANOVA
    from scipy.stats import f_oneway
    
    wide_data = [col.dropna() for _, col in
                 long_data.pivot(index=None,
                                 columns="dependent_var_name",
                                 values="independent_var_name").iteritems()]
    
    anova = pd.DataFrame.from_dict({
        "dependent_var_modality": "overall_margin",
        "indicator": "anova_oneway",
        "value": f_oneway(*wide_data).pvalue},
        orient="index",
    ).transpose()
    #############
    
    ########## MODEL
    dependent_var_cat = dependent_var.astype(CategoricalDtype(ordered=False))
    ref_modality_dependent = groupby_stats.loc[lambda df: df["indicator"] == "nonNA_count", :] \
                                 .loc[lambda df: df["value"] == df["value"].max(), :] \
                                 .iloc[0, :] \
        ["dependent_var_modality"]
    new_levels = [ref_modality_dependent] + pd.CategoricalIndex(dependent_var_cat) \
        .remove_categories(ref_modality_dependent).categories.tolist()
    dependent_var_cat.cat.reorder_categories(new_levels, inplace=True)
    
    X = independent_var.rename(independent_var_name).to_frame().assign(intercept=1)
    model = MNLogit(dependent_var_cat, X)
    results = model.fit()
    
    params = results.params
    params.columns = dependent_var_cat.cat.categories[1:]
    params = params.rename_axis("dependent_var_modality", axis=1) \
        .rename_axis("independent_var", axis=0) \
        .drop("intercept", axis=0) \
        .melt(var_name="dependent_var_modality") \
        .assign(indicator="coeffs_LogR")
    
    ########### LRT
    LRT = pd.DataFrame.from_dict({
        "dependent_var_modality": "overall_margin",
        "indicator": "pvalue_LRT_LogR",
        "value": results.llr_pvalue
    }, orient="index").transpose()
    
    ########## pvalues model
    pvalues = results.pvalues
    pvalues.columns = dependent_var_cat.cat.categories[1:]
    pvalues = pvalues.rename_axis("independent_var_name") \
        .drop("intercept", axis=0) \
        .reset_index(drop=False) \
        .melt(id_vars="independent_var_name",
              var_name="dependent_var_modality") \
        .assign(indicator="pvalue_coeff_LogR") \
        .drop("independent_var_name", axis=1)
    
    ####### conf int param model
    conf_int = results.conf_int() \
                   .reset_index(level=1, drop=False) \
                   .rename({"level_1": "independent_var_name"}, axis=1) \
                   .rename_axis("dependent_var_modality", axis=0) \
                   .loc[lambda df: df["independent_var_name"] != "intercept", :] \
        .reset_index(drop=False) \
        .rename({"lower": "coeff_LogR_lb", "upper": "coeff_LogR_ub"}, axis=1) \
        .melt(id_vars=["dependent_var_modality", "independent_var_name"],
              value_vars=["coeff_LogR_lb", "coeff_LogR_ub"],
              var_name="indicator") \
        .drop("independent_var_name", axis=1)
    multicategorical_continuous = pd.concat([groupby_stats, params, pvalues, conf_int, LRT, anova], axis=0) \
                                 .assign(ref_modality_dependent=ref_modality_dependent,
                                         ref_modality_independent=np.NaN,
                                         independent_var_modality=np.NaN,
                                         independent_var_name=independent_var_name,
                                         dependent_var_name=dependent_var_name)
    return multicategorical_continuous[list_columns_names_export]


def multicategorical_multicategorical(dependent_var:pd.Series,
                                      independent_var:pd.Series,
                                      list_columns_names_export: list):
    dependent_var_name = dependent_var.name
    independent_var_name = independent_var.name
    crosscount = pd.crosstab(index=dependent_var,
                             columns=independent_var,
                             margins=True,
                             margins_name="overall_margin")
    ref_modality_independent = crosscount.transpose() \
                                   .loc[
                               lambda df: df["overall_margin"] == df["overall_margin"].drop("overall_margin").max(), :] \
        .index[0]
    independent_dummies = pd.get_dummies(independent_var, drop_first=False) \
        .drop(ref_modality_independent, axis=1) \
        .assign(intercept=1)
    
    dependent_var_cat = dependent_var.astype(CategoricalDtype(ordered=False))
    ref_modality_dependent = \
    crosscount.loc[lambda df: df["overall_margin"] == df["overall_margin"].drop("overall_margin").max(), :] \
        .index[0]
    new_levels = [ref_modality_dependent] + pd.CategoricalIndex(dependent_var_cat) \
        .remove_categories(ref_modality_dependent) \
        .categories.tolist()
    dependent_var_cat.cat.reorder_categories(new_levels, inplace=True)
    
    # MODEL
    # model = Logit(dependent_var_cat.cat.codes, independent_dummies)
    model = MNLogit(dependent_var_cat, independent_dummies)
    results = model.fit()
    
    params = results.params
    params.columns = dependent_var_cat.cat.categories[1:]
    params = params.rename_axis("dependent_var_modality") \
        .drop("intercept", axis=0) \
        .reset_index(drop=False) \
        .melt(id_vars="dependent_var_modality",
              var_name="independent_var_modality") \
        .assign(indicator="coeffs_LogR")
    
    pvalues = results.pvalues
    pvalues.columns = dependent_var_cat.cat.categories[1:]
    pvalues = pvalues.rename_axis("dependent_var_modality") \
        .drop("intercept", axis=0) \
        .reset_index(drop=False) \
        .melt(id_vars="dependent_var_modality",
              var_name="independent_var_modality") \
        .assign(indicator="pvalue_coeff_LogR") \
        .round({"independent_var_modality": 2})
    
    conf_int = results.conf_int() \
                   .reset_index(level=1, drop=False) \
                   .rename({"level_1": "independent_var_modality"}, axis=1) \
                   .loc[lambda df: df["independent_var_modality"] != "intercept", :] \
        .rename_axis("dependent_var_modality") \
        .reset_index(drop=False) \
        .rename({"lower": "coeff_LogR_lb",
                 "upper": "coeff_LogR_ub"},
                axis=1) \
        .melt(id_vars=["dependent_var_modality", "independent_var_modality"],
              value_vars=["coeff_LogR_lb", "coeff_LogR_ub"],
              var_name="indicator")
    
    LRT = pd.DataFrame.from_dict({
        "dependent_var_modality": "overall_margin",
        "independent_var_modality": "overall_margin",
        "indicator": "pvalue_LRT_LogR",
        "value": results.llr_pvalue},
        orient="index",
    ).transpose()
    chi2_pvalue = chi2_contingency(crosscount.drop("overall_margin", axis=0) \
                                   .drop("overall_margin", axis=1))[1]
    
    chi2_crosscount = pd.DataFrame.from_dict({
        "dependent_var_modality": "overall_margin",
        "independent_var_modality": "overall_margin",
        "indicator": "pvalue_chisquare_crosscount",
        "value": chi2_pvalue},
        orient="index",
    ).transpose()
    multicategorical_multicategorical = pd.concat([params, pvalues, conf_int, LRT, chi2_crosscount], axis=0) \
        .assign(ref_modality_dependent=ref_modality_dependent,
                ref_modality_independent=ref_modality_independent,
                independent_var_name=independent_var_name,
                dependent_var_name=dependent_var_name)
    return multicategorical_multicategorical[list_columns_names_export]


def continuous_multicategorical(dependent_var, independent_var):

    dependent_var_name = dependent_var.name
    independent_var_name = independent_var.name
    long_data = pd.DataFrame.from_dict({"dependent_var_name": dependent_var,
                                        "independent_var_name": independent_var},
                                       orient="columns")

    groupby_stats = long_data \
        .groupby("independent_var_name") \
        .agg(["min", "max", "median", "mean", "count"]) \
        .droplevel(level=0, axis=1) \
        .rename({"count": "nonNA_count"}, axis=1) \
        .reset_index(drop=False) \
        .melt(id_vars="independent_var_name",
              var_name="indicator") \
        .rename({"independent_var_name": "independent_var_modality"},
                axis=1) \
        .round({"value": 2})
    ###############

    ############# ANOVA

    wide_data = [col.dropna() for _, col in
                 long_data.pivot(index=None,
                                 columns="independent_var_name",
                                 values="dependent_var_name").iteritems()]

    anova = pd.DataFrame.from_dict({
        "dependent_var_modality": "overall_margin",
        "indicator": "anova_oneway",
        "value": f_oneway(*wide_data).pvalue},
        orient="index",
    ).transpose()
    #############

    
    
    ########## MODEL
    ref_modality_independent = groupby_stats\
                               .loc[lambda df: df["indicator"] == "nonNA_count", :]\
                               .loc[lambda df: df["value"] == df["value"].max(), :]\
                               .iloc[0, :]\
                               ["independent_var_modality"]
    
    independent_dummies = pd.get_dummies(independent_var, drop_first=False) \
        .drop(ref_modality_independent, axis=1) \
        .assign(intercept=1)

    import statsmodels.api as sm
    
    model = sm.GLM(dependent_var, independent_dummies, family=sm.families.Binomial())
    results = model.fit()

    params = results.params
    params = params\
        .rename_axis("independent_var_modality", axis=0) \
        .drop("intercept", axis=0) \
        .reset_index(drop=False) \
        .rename_axis()\
        .rename({0: "value"}, axis=1)\
        .assign(indicator="coeffs_LogR")

    ########### LRT
    LRT = pd.DataFrame.from_dict({
        "dependent_var_modality": "overall_margin",
        "indicator": "pvalue_LRT_LogR",
        "value": results.llr_pvalue
    }, orient="index").transpose()

    ########## pvalues model
    pvalues = results.pvalues
    pvalues.columns = dependent_var_cat.cat.categories[1:]
    pvalues = pvalues.rename_axis("independent_var_name") \
        .drop("intercept", axis=0) \
        .reset_index(drop=False) \
        .melt(id_vars="independent_var_name",
              var_name="dependent_var_modality") \
        .assign(indicator="pvalue_coeff_LogR") \
        .drop("independent_var_name", axis=1)

    ####### conf int param model
    conf_int = results.conf_int() \
                   .reset_index(level=1, drop=False) \
                   .rename({"level_1": "independent_var_name"}, axis=1) \
                   .rename_axis("dependent_var_modality", axis=0) \
                   .loc[lambda df: df["independent_var_name"] != "intercept", :] \
        .reset_index(drop=False) \
        .rename({"lower": "coeff_LogR_lb", "upper": "coeff_LogR_ub"}, axis=1) \
        .melt(id_vars=["dependent_var_modality", "independent_var_name"],
              value_vars=["coeff_LogR_lb", "coeff_LogR_ub"],
              var_name="indicator") \
        .drop("independent_var_name", axis=1)
    multicategorical_continuous = pd.concat([groupby_stats, params, pvalues, conf_int, LRT, anova], axis=0) \
        .assign(ref_modality_dependent=ref_modality_dependent,
                ref_modality_independent=np.NaN,
                independent_var_modality=np.NaN,
                independent_var_name=independent_var_name,
                dependent_var_name=dependent_var_name)
    
    
    return

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

