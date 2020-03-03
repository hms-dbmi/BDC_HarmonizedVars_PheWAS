from datetime import datetime

import pandas as pd
import numpy as np

from python_lib.utils import timer_decorator


@timer_decorator
def _is_not_number(s):
    try:
        float(s)
    except:
        return True
    return False


@timer_decorator
def quality_checking(study_df: pd.DataFrame) -> np.array:
    """
    1. Assert that all numerical columns are really numericals
    2. Filter the input DataFrame regarding two criteria:
    - Drops columns with ID or identifier in their names
    - Drops columns with 1 or less non-null values
    - Drops columns of type strings with only unique values (identifier, cannot be used in a stat analysis anyway)
    :param study_df:
    :return:
    """

    def _filter_unique_values(study_df) -> np.array:
        """
        Returns an array of boolean values, with True for the column that
        are strings with only one single unique (or missing) values
        """
        study_df = study_df.apply(pd.to_numeric, errors="ignore")
        mask_strings = study_df.dtypes == "O"
        unique_df = study_df.loc[:, mask_strings]
        only_unique_values = unique_df.apply(lambda x: len(x.dropna().unique()) == len(x.dropna()))
        name_unique_values_cols = only_unique_values.where(only_unique_values == True).dropna().index
        return study_df.columns.isin(name_unique_values_cols)
    
    import warnings
    warnings.filterwarnings("ignore", 'This pattern has match groups')
    mask_varnames = study_df.columns.str.contains("((\\\\|_|^|\W)ID(\\\\|_|\W|$))|(identifier)", regex=True)
    zeroOne_filter = study_df.apply(lambda x: len(x.dropna().unique()) in [0, 1])
    unique_filter = _filter_unique_values(study_df)
    
    var_name_to_drop = study_df.loc[:, (mask_varnames | zeroOne_filter | unique_filter)].columns
    print("{0} variables to drop after quality checking".format(len(var_name_to_drop)))
    return var_name_to_drop


@timer_decorator
def inferring_categorical_columns(study_df: pd.DataFrame,
                                  threshold_categorical=4) -> pd.DataFrame:
    """
    Infer categorical types for numerical variables if they match two criteria:
    - Only containing integers
    - 
    :param study_df:
    :param threshold_categorical:
    :return:g
    """
    float_df = study_df.loc[:, study_df.dtypes == "float"]
    int_columns = study_df.loc[:, study_df.dtypes == "int"].columns.tolist()
    col_is_actually_int = [name for name, col in float_df.iteritems() if
                           col.dropna().apply(lambda x: x.is_integer()).all()]
    int_columns = int_columns + col_is_actually_int
    int_df = study_df.loc[:, int_columns]
    unique_values = int_df.apply(lambda x: len(x.unique()))
    potential_categorical = int_df.loc[:, unique_values.values <= threshold_categorical]\
        .columns
    return potential_categorical


@timer_decorator
def quality_filtering(study_df):
    inferred_cat_columns = inferring_categorical_columns(study_df)
    study_df[inferred_cat_columns] = study_df[inferred_cat_columns].astype(str)
    variables_to_drop = quality_checking(study_df)
    filtered_df = study_df.drop(variables_to_drop, axis=1)
    return filtered_df


@timer_decorator
def get_study_variables_info(original_df: pd.DataFrame,
                      filtered_df) -> dict:
    total_nb_subjects, nb_variables = original_df.shape
    nb_variables_dic = {
        "Nb total variables": nb_variables, 
        "Total number subjects": total_nb_subjects,
        "ID variables": original_df.shape[1] - filtered_df.shape[1],
        "Phenotypes variables": filtered_df.shape[1]
    }
    
    non_null_values = filtered_df.notnull().sum()
    prop_non_null_values = non_null_values.to_frame() / filtered_df.shape[0]
    thresholds = {
        "nb var > 10% non-null values": 0.1, 
        "nb var > 25% non-null values": 0.25, 
        "nb var > 50% non-null values": 0.5, 
        "nb var > 75% non-null values": 0.75,
        "nb var > 90% non-null values": 0.9
    }
    dic_quantiles = {k: prop_non_null_values.apply(lambda x: x > threshold).sum().values[0] for k, threshold in thresholds.items()}
    
    mean_non_null = round(non_null_values.mean(),1)
    median_non_null = non_null_values.median()
    non_null = {**dic_quantiles, 
               "Mean non-null value count per variable": mean_non_null,
               "Median non-null value count per variable": median_non_null
    }
    
    var_dtypes = filtered_df.dtypes.value_counts().to_dict()
    long_dictionary = {
        "Variables count": nb_variables_dic, 
        "Number variables with non-null values": non_null,
        "Variable types": var_dtypes
    }
    return long_dictionary


def _variables_description(study_df: pd.DataFrame):
    categorical_describe = study_df.describe(include=['object']).transpose()
    numerical_describe = study_df.describe(include=["float", "int"]).transpose()
    return 



def pandas_profiling_test(df):
    from pandas_profiling import ProfileReport
    
    report = ProfileReport(df, minimal=True)
    
    description = report.get_description()
    
    table = description["table"]
    
    return table


def table_one():
    
    return

