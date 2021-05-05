def get_study_info(original_df: pd.DataFrame,
                   filtered_df) -> dict:
    total_nb_subjects, nb_variables = original_df.shape
    nb_variables_dic = {
        "Nb total variables": nb_variables,
        "Total number subjects": total_nb_subjects,
        "ID variables": original_df.shape[1] - filtered_df.shape[1],
        "Phenotypes variables": filtered_df.shape[1]
    }
    
    non_null_values = filtered_df.notnull().sum()
    #
    
    # prop_non_null_values = non_null_values.to_frame() / filtered_df.shape[0]
    # thresholds = {
    #     "nb var > 10% non-null values": 0.1,
    #     "nb var > 25% non-null values": 0.25,
    #     "nb var > 50% non-null values": 0.5,
    #     "nb var > 75% non-null values": 0.75,
    #     "nb var > 90% non-null values": 0.9
    # }
    # dic_quantiles = {k: prop_non_null_values.apply(lambda x: x > threshold).sum().values[0] for k, threshold in thresholds.items()}
    
    # mean_non_null = round(non_null_values.mean(),1)
    # median_non_null = non_null_values.median()
    # non_null = {**dic_quantiles,
    #            "Mean non-null value count per variable": mean_non_null,
    #            "Median non-null value count per variable": median_non_null
    # }
    
    # var_dtypes = filtered_df.dtypes.value_counts().to_dict()
    # long_dictionary = {
    #     "Variables count": nb_variables_dic,
    #     "Number variables with non-null values": non_null,
    #     "Variable types": var_dtypes
    # }
    return long_dictionary


def _variables_description(study_df: pd.DataFrame):
    categorical_describe = study_df.describe(include=['object']).transpose()
    numerical_describe = study_df.describe(include=["float", "int"]).transpose()
    return