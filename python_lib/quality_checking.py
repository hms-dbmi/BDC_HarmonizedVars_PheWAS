from datetime import datetime

import pandas as pd
import numpy as np


def _is_not_number(s):
    try:
        float(s)
    except:
        return True
    return False


def _replace_null_values(study_df):
    return study_df.replace(["N/A", "null", "NULL", "Null"], np.NaN)



def _quality_checking(study_df: pd.DataFrame,
                      threshold_nonnull_values: int) -> np.array:
    """
    1. Assert that all numerical columns are really numericals
    2. Filter the input DataFrame regarding two criteria:
    - Drops columns with ID or identifier in their names
    - Drops columns with equal or less non-null values than specified by threshold
    - Drops columns of type strings with only unique values (identifier, cannot be used in a stat analysis anyway)
    :param threshold_nonnull_values:
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
    list_masks = []
    list_patterns = [r"(?<=[\W]|^)ID(?=\W|$)", "identifier", "consent"]
    for pattern in list_patterns:
        list_masks.append(study_df.columns.str.contains(pattern, regex=True, case=False).tolist())
    mask_varnames = pd.DataFrame(list_masks).any()
    nonnull_filter = study_df.apply(lambda x: len(x.dropna()) <= threshold_nonnull_values)
    unique_filter = _filter_unique_values(study_df)
    
    var_name_to_drop = study_df.loc[:, (mask_varnames.values | nonnull_filter.values | unique_filter)].columns
    print("{0} variables to drop after quality checking".format(len(var_name_to_drop)))
    return var_name_to_drop


def _inferring_categorical_columns(study_df: pd.DataFrame,
                                   threshold_categorical=4) -> pd.DataFrame:
    """
    Infer categorical types for numerical variables if they match two criteria:
    - Only containing integers
    - Less or equal numbers of unique values as specified by threshold_categorical
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


def quality_filtering(study_df, threshold_nonnull_values):
    study_df = _replace_null_values(study_df)
    inferred_cat_columns = _inferring_categorical_columns(study_df)
    study_df[inferred_cat_columns] = study_df[inferred_cat_columns].astype(str)
    variables_to_drop = _quality_checking(study_df, threshold_nonnull_values)
    filtered_df = study_df.drop(variables_to_drop, axis=1)
    return filtered_df
