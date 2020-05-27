from datetime import datetime

import pandas as pd
import numpy as np 
from scipy import stats

import PicSureHpdsLib
import PicSureClient


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
        mask_modalities = subset_variablesDict["categorical"] == False | subset_variablesDict["nb_modalities"].between(*nb_categories, inclusive=True)
        subset_variablesDict = subset_variablesDict.loc[mask_modalities, :]

    return subset_variablesDict


def _LRT(dependent_var_name: str,
        independent_var_names: str,
        study_df: pd.DataFrame) -> dict:
    
    from statsmodels.discrete.discrete_model import Logit
    from scipy.linalg import LinAlgError
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    from tqdm import tqdm
    
    dic_pvalues = {}
    dic_errors = {}
    for independent_var_name in tqdm(independent_var_names, position=0, leave=True):
        subset_df = study_df.loc[:, [dependent_var_name, independent_var_name]]\
                  .dropna(how="any")
        
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
            params = results.params
            conf = np.exp(results.conf_int())
            conf['Odds Ratio'] = np.exp(params)
            conf.columns = ['5%', '95%', 'Odds Ratio']
            dic_pvalues[independent_var_name] = (results.llr_pvalue, conf)
        except (LinAlgError, PerfectSeparationError) as e:
            dic_pvalues[independent_var_name] = np.NaN
            #dic_errors[independent_var_name] = e
    return dic_pvalues, dic_errors


def PheWAS(study_variables: list,
           dependent_var_name: str,
           resource) -> dict:
    from python_lib.wrappers import query_runner
    vars_to_query = study_variables + [dependent_var_name]
    study_df = query_runner(resource=resource,
                 to_anyof=vars_to_query,
                 result_type="DataFrame",
                 low_memory=False, 
                 timeout=500)
    print("Shape of retrieved HPDS dataframe {0}".format(study_df.shape))        
    study_df[dependent_var_name] = study_df[dependent_var_name].astype("category")
    dic_pvalues, dic_errors = _LRT(dependent_var_name, 
                       study_variables,
                       study_df)
    return dic_pvalues, dic_errors

