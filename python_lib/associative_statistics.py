
import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.regression.linear_model import OLS
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr


# TODO: handling the situations where value count inf to 5, discard variables automatically

list_columns_names_export = [
    "dependent_var_modality",
    "ref_modality_dependent",
    "independent_var_modality",
    "ref_modality_independent",
    "indicator",
    "value"
]


class EndogOrExogUnique(Exception):
    """Raised when either the dependent or the independent variable are
             all NaN after removing NaN"""


class CrossCountThresholdError(Exception):
    """Raised when the marginal crosscount of non null values is below the
             specified value in parameters"""


def nonNA_crosscount(df, dependent_var_name, independent_var_names):
    dependent_not_null = df[dependent_var_name].notnull()
    independent_not_null = df[independent_var_names].notnull()
    non_na_sum = independent_not_null.apply(lambda serie: pd.concat([dependent_not_null, serie], axis=1) \
                                            .all(axis=1)) \
        .sum() \
        .rename("value") \
        .rename_axis("independent_var_name", axis=0) \
        .to_frame() \
        .assign(indicator="nonNA_crosscount",
                dependent_var_modality="overall_margin",
                independent_var_modality="overall_margin")
    return non_na_sum


class associationStatistics():
    
    def __init__(self,
                 dependent_var: pd.Series,
                 independent_var: pd.Series):
        self.dependent_var = dependent_var
        self.independent_var = independent_var
        self.ref_modality_dependent = None
        self.ref_modality_independent = None
        self.list_df_results_statistics = {}
        self.logs = []
    
    def drop_na(self):
        mask_not_null = pd.concat([self.dependent_var, self.independent_var], axis=1).notnull().all(axis=1)
        self.dependent_var = self.dependent_var[mask_not_null]
        self.independent_var = self.independent_var[mask_not_null]
        
    
    def normalize_continuous_var(self):
        
        def _normalize_var(var):
            return 2*(var - min(var))/(max(var) - min(var)) - 1
        if not hasattr(self.independent_var, "cat"):
            self.independent_var = _normalize_var(self.independent_var)
        
        
    def recode_levels(self):
        
        def _get_ref_modality(variable):
            ref_modality = variable.value_counts() \
                .loc[lambda serie: serie == serie.max()] \
                .index[0]
            return ref_modality
        
        def _change_ref_modality_variable(variable, ref_modality):
            new_levels = [ref_modality] + pd.CategoricalIndex(variable) \
                .remove_categories(ref_modality) \
                .categories.tolist()
            variable.cat = pd.CategoricalIndex(variable).reorder_categories(new_levels)
            return variable
        if hasattr(self.independent_var, 'cat'):
            self.independent_var = self.independent_var.cat.remove_unused_categories()
            self.ref_modality_independent = _get_ref_modality(self.independent_var)
            self.independent_var = _change_ref_modality_variable(self.independent_var, self.ref_modality_independent)
        if hasattr(self.dependent_var, 'cat'):
            self.dependent_var = self.dependent_var.cat.remove_unused_categories()
            self.ref_modality_dependent = _get_ref_modality(self.dependent_var)
            self.dependent_var = _change_ref_modality_variable(self.dependent_var, self.ref_modality_dependent)
        return
        
    
    
    def groupby_statistics(self):
        if hasattr(self.dependent_var, 'cat'):
            if hasattr(self.independent_var, 'cat'):
                groupby_stats = pd.crosstab(index=self.dependent_var,
                                         columns=self.independent_var,
                                         margins=True,
                                         margins_name="overall_margin") \
                    .rename_axis("dependent_var_modality", axis=0) \
                    .reset_index(drop=False) \
                    .melt(id_vars="dependent_var_modality",
                          var_name="independent_var_modality") \
                    .assign(indicator="nonNA_crosscount")
            else:
                groupby_stats = pd.DataFrame.from_dict({"dependent_var_name": self.dependent_var,
                                            "independent_var_name": self.independent_var},
                                           orient="columns") \
                    .groupby("dependent_var_name") \
                    .agg(["min", "max", "median", "mean", "count"]) \
                    .droplevel(level=0, axis=1) \
                    .rename({"count": "nonNA_crosscount"}, axis=1) \
                    .reset_index(drop=False) \
                    .melt(id_vars="dependent_var_name",
                          var_name="indicator") \
                    .rename({"dependent_var_name": "dependent_var_modality"},
                            axis=1)
                self.list_df_results_statistics["groupby_stats"] = groupby_stats
        else:
            if hasattr(self.independent_var, 'cat'):
                groupby_stats = pd.DataFrame.from_dict({
                        "dependent_var": self.dependent_var,
                        "independent_var": self.independent_var},
                        orient="columns") \
                    .groupby("independent_var") \
                    .agg(["min", "max", "median", "mean", "count"]) \
                    .droplevel(level=0, axis=1) \
                    .rename({"count": "nonNA_crosscount"}, axis=1) \
                    .reset_index(drop=False) \
                    .melt(id_vars="independent_var",
                          var_name="indicator") \
                    .rename({"independent_var": "independent_var_modality"},
                            axis=1)
            else:
                groupby_stats = self.dependent_var\
                    .agg(["min", "max", "std", "mean", "median", "count"])\
                    .rename("value") \
                    .rename_axis("indicator", axis=0)\
                    .reset_index()\
                    .replace({"indicator": {"count": "nonNA_crosscount"}})
        self.list_df_results_statistics["groupby_stats"] = groupby_stats
    
    
    def test_statistics(self):
        if (len(self.dependent_var.dropna().unique()) <= 1) | (len(self.independent_var.dropna().unique()) <= 1):
            raise EndogOrExogUnique
        if hasattr(self.dependent_var, "cat"):
            if hasattr(self.dependent_var, "cat"):
                chi2_pvalue = chi2_contingency(pd.crosstab(index=self.dependent_var,
                                                           columns=self.independent_var,
                                                           margins=True,
                                                           margins_name="overall_margin") \
                                               .drop("overall_margin", axis=0) \
                                               .drop("overall_margin", axis=1))[1]
                self.list_df_results_statistics["chi2_crosscount"] = pd.DataFrame.from_dict({
                    "dependent_var_modality": "overall_margin",
                    "independent_var_modality": "overall_margin",
                    "indicator": "pvalue_chisquare_crosscount",
                    "value": chi2_pvalue},
                    orient="index",
                ).transpose()
            else:
                wide_data = pd.DataFrame.from_dict({"dependent_var": self.dependent_var,
                                                     "independent_var": self.independent_var},
                                                    orient="columns").pivot(index=None,
                                                                            columns="dependent_var",
                                                                            values="independent_var")
                anova_data = [col.dropna() for _, col in wide_data.iteritems()]
                self.list_df_results_statistics["anova"] = pd.DataFrame.from_dict({
                    "dependent_var_modality": "overall_margin",
                    "indicator": "pvalue_anova_oneway",
                    "value": f_oneway(*anova_data).pvalue},
                    orient="index",
                ).transpose()

        else:
            if hasattr(self.independent_var, "cat"):
                wide_data = pd.DataFrame.from_dict({"dependent_var": self.dependent_var,
                                                     "independent_var": self.independent_var},
                                                    orient="columns").pivot(index=None,
                                                                            columns="independent_var",
                                                                            values="dependent_var")
                anova_data = [col.dropna() for _, col in wide_data.iteritems()]
                self.list_df_results_statistics["anova"] = pd.DataFrame.from_dict({
                    "independent_var_modality": "overall_margin",
                    "indicator": "pvalue_anova_oneway",
                    "value": f_oneway(*anova_data).pvalue},
                    orient="index",
                ).transpose()
            else:
                self.list_df_results_statistics["descriptive_stats"] = self.dependent_var \
                    .loc[self.independent_var.notnull()]\
                    .agg(["min", "max", "std", "mean", "median", "count"]) \
                    .rename("value") \
                    .rename_axis("indicator", axis=0) \
                    .reset_index()
                pearson = pearsonr(self.dependent_var, self.independent_var)
                # noinspection PyTypeChecker
                spearman = spearmanr(self.dependent_var, self.independent_var)
                self.list_df_results_statistics["correlation"] = pd.Series({
                    "spearman": spearman[0],
                    "pearson": pearson[0],
                    "pvalue_pearson": pearson[1],
                    "pvalue_spearman": spearman[1]
                }).reset_index(drop=False) \
                    .rename({"index": "indicator", 0: "value"}, axis=1)
        return
    
    def create_model(self):
        if (len(self.dependent_var.dropna().unique()) <= 1) | (len(self.independent_var.dropna().unique()) <= 1):
            raise EndogOrExogUnique
        if hasattr(self.independent_var, 'cat'):
            X = pd.get_dummies(self.independent_var.astype(str), drop_first=False) \
                .drop(self.ref_modality_independent, axis=1) \
                .assign(intercept=1)
        else:
            X = self.independent_var.to_frame().assign(intercept=1)
        if hasattr(self.dependent_var, 'cat'):
            y = self.dependent_var.cat.codes
            return MNLogit(y, X)
        else:
            y = self.dependent_var.values
            return OLS(y, X)
        
        
    def model_results_handling(self, results):
        params = results.params
        pvalues = results.pvalues
        if hasattr(self.dependent_var, "cat"):
            dependent_variable_modalities = self.dependent_var.cat.categories.drop(self.ref_modality_dependent)
            params.columns = dependent_variable_modalities
            pvalues.columns = dependent_variable_modalities
            if hasattr(self.independent_var, "cat"):
                self.list_df_results_statistics["params"] = params.rename_axis("dependent_var_modality") \
                    .drop("intercept", axis=0) \
                    .reset_index(drop=False) \
                    .melt(id_vars="dependent_var_modality",
                          var_name="independent_var_modality") \
                    .assign(indicator="coeff_LogR")
                self.list_df_results_statistics["pvalues"] = pvalues.rename_axis("dependent_var_modality", axis=0) \
                    .drop("intercept", axis=0) \
                    .reset_index(drop=False) \
                    .melt(id_vars="dependent_var_modality",
                          var_name="independent_var_modality") \
                    .assign(indicator="pvalue_coeff_LogR")
                self.list_df_results_statistics["conf_int"] = results.conf_int() \
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
                self.list_df_results_statistics["LRT"] = pd.DataFrame.from_dict({
                    "dependent_var_modality": "overall_margin",
                    "independent_var_modality": "overall_margin",
                    "indicator": "pvalue_LRT_LogR",
                    "value": results.llr_pvalue},
                    orient="index",
                ).transpose()
            else:
                self.list_df_results_statistics["params"] = params.rename_axis("dependent_var_modality", axis=1) \
                    .rename_axis("independent_var", axis=0) \
                    .drop("intercept", axis=0) \
                    .melt(var_name="dependent_var_modality") \
                    .assign(indicator="coeff_LogR")
                ########## pvalues model
                self.list_df_results_statistics["pvalues"] = pvalues.rename_axis("independent_var_name") \
                    .drop("intercept", axis=0) \
                    .reset_index(drop=False) \
                    .melt(id_vars="independent_var_name",
                          var_name="dependent_var_modality") \
                    .assign(indicator="pvalue_coeff_LogR") \
                    .drop("independent_var_name", axis=1)
                ####### conf int param model
                self.list_df_results_statistics["conf_int"] = results.conf_int() \
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
                ########### LRT
                self.list_df_results_statistics["LRT"] = pd.DataFrame.from_dict({
                    "dependent_var_modality": "overall_margin",
                    "indicator": "pvalue_LRT_LogR",
                    "value": results.llr_pvalue
                }, orient="index").transpose()
        else:
            llr_pvalue = results.compare_lr_test(OLS(results.model.endog, np.ones_like(results.model.endog)).fit())[1]
            if hasattr(self.independent_var, "cat"):
                ########## MODEL
                self.list_df_results_statistics["params"] = params \
                    .rename_axis("independent_var_modality", axis=0) \
                    .drop("intercept", axis=0) \
                    .reset_index(drop=False) \
                    .rename({0: "value"}, axis=1) \
                    .assign(indicator="coeff_LR")
                ########### LRT
                self.list_df_results_statistics["LRT"] = pd.DataFrame.from_dict({
                    "independent_var_modality": "overall_margin",
                    "indicator": "pvalue_LRT_LR",
                    "value": llr_pvalue
                }, orient="index") \
                    .transpose()
                ########## pvalues model
                self.list_df_results_statistics["pvalues"] = pvalues.rename_axis("independent_var_modality") \
                    .rename("value") \
                    .drop("intercept", axis=0) \
                    .reset_index(drop=False) \
                    .assign(indicator="pvalue_coeff_LR")
                ####### conf int param model
                self.list_df_results_statistics["conf_int"] = results.conf_int() \
                                                                  .reset_index(drop=False) \
                                                                  .rename({"index": "independent_var_modality",
                                                                           0: "coeff_LR_lb",
                                                                           1: "coeff_LR_ub"}, axis=1) \
                                                                  .loc[lambda df: df["independent_var_modality"] != "intercept", :] \
                    .melt(id_vars="independent_var_modality",
                          value_vars=["coeff_LR_lb", "coeff_LR_ub"],
                          var_name="indicator")
            else:
                self.list_df_results_statistics["LRT"] = pd.DataFrame.from_dict({
                    "independent_var_modality": "overall_margin",
                    "indicator": "pvalue_LRT_LR",
                    "value": llr_pvalue
                }, orient="index") \
                    .transpose()
                self.list_df_results_statistics["params"] = results.params \
                    .to_frame() \
                    .drop("intercept") \
                    .rename({0: "value"}, axis=1) \
                    .assign(indicator="coeff_LR") \
                    .reset_index(drop=True)
                self.list_df_results_statistics["conf_int"] = results.conf_int() \
                    .drop("intercept", axis=0) \
                    .rename({0: "coeff_LR_lb",
                             1: "coeff_LR_ub"}, axis=1) \
                    .melt(value_vars=["coeff_LR_lb", "coeff_LR_ub"],
                          var_name="indicator")
                self.list_df_results_statistics["pvalues"] = results.pvalues \
                    .drop("intercept") \
                    .to_frame() \
                    .rename({0: "value"}, axis=1) \
                    .assign(indicator="pvalue_coeff_LR") \
                    .reset_index(drop=True)
        return

    def creating_empty_df_results(self):
        if hasattr(self.dependent_var, 'cat'):
            if hasattr(self.independent_var, 'cat'):
                list_indicators = [
                    "pvalue_LRT_LogR",
                    "coeff_LogR",
                    "coeff_LogR_ub",
                    "coeff_LogR_lb",
                    "pvalue_coeff_LogR",
                    "pvalue_chisquare_crosscount",
                ]
                null_results = pd.DataFrame.from_dict({
                    "indicator": list_indicators,
                    "value": np.NaN,
                    "dependent_var_modality": np.NaN,
                    "independent_var_modality": np.NaN,
                })
            else:
                list_indicators = [
                    "pvalue_LRT_LogR",
                    "coeff_LogR",
                    "coeff_LogR_ub",
                    "coeff_LogR_lb",
                    "pvalue_coeff_LogR",
                    "pvalue_anova_oneway"
                ]
                null_results = pd.DataFrame.from_dict({
                    "indicator": list_indicators,
                    "value": np.NaN,
                    "dependent_var_modality": np.NaN,
                })
        else:
            if hasattr(self.independent_var, 'cat'):
                list_indicators = [
                    "pvalue_anova_oneway"
                    "coeff_LR",
                    "pvalue_coeff_LR",
                    "coeff_LR_lb",
                    "coeff_LR_ub",
                    "pvalue_LRT_LR",
                ]
                null_results = pd.DataFrame.from_dict({
                    "indicator": list_indicators,
                    "value": np.NaN,
                    "independent_var_modality": np.NaN,
                })
            else:
                list_indicators = [
                    "spearman",
                    "pearson",
                    "pvalue_spearman",
                    "pvalue_pearson",
                    "coeff_LR",
                    "pvalue_coeff_LR",
                    "coeff_LR_lb",
                    "coeff_LR_ub"
                    "pvalue_LRT_LR",
                ]
                null_results = pd.DataFrame.from_dict({
                    "indicator": list_indicators,
                    "value": np.NaN,
                })
        self.list_df_results_statistics["null_results"] = null_results
        return

    def gathering_statistics(self):
        statistics = pd.concat([v for v in self.list_df_results_statistics.values()],
                               axis=0,
                               ignore_index=True) \
            .assign(ref_modality_dependent=self.ref_modality_dependent,
                    ref_modality_independent=self.ref_modality_independent) # Assigning None for continuous variables
        if hasattr(self.dependent_var, "cat"):
            if not hasattr(self.independent_var, "cat"):
                statistics = statistics.assign(independent_var_modality=np.NaN)
        else:
            if hasattr(self.independent_var, "cat"):
                statistics = statistics.assign(dependent_var_modality=np.NaN)
            else:
                statistics = statistics.assign(dependent_var_modality=np.NaN,
                                               independent_var_modality=np.NaN)
        return statistics[list_columns_names_export]
    
if __name__ == '__main__':
    var_test = "\\DCC Harmonized data set\\04 - Blood cell count\\Measurement of the ratio of variation in width to the mean width of the red blood cell (rbc) volume distribution curve taken at +/- 1 CV  known as red cell distribution width (RDW).\\"
    ind_test = "\\Cardiovascular Health Study (CHS) Cohort: an NHLBI-funded observational study of risk factors for cardiovascular disease in adults 65 years or older ( phs000287 )\\Brief up-date on general health status and changes that may have occurred since last visit. Data were collected at year 17 (6-months phone interview follow-up).\\DAYS IN BED FOR ILLNESS OR INJURY\\"
    
    instance = associationStatistics(var_test, ind_test)
    
