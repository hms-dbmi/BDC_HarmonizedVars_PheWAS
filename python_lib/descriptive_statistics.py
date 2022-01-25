import pandas as pd
import numpy as np


def get_descriptive_statistics(filtered_df, filtered_independent_var_names):
    def _count_modalities(var_name, serie):
        value_counts = serie.value_counts() \
            .rename("value")
        value_counts["total"] = value_counts.sum()
        value_counts = value_counts.to_frame() \
            .assign(var_name=var_name, var_type=lambda df: "multicategorical" if len(df["value"]) > 3 else "binary") \
            .rename_axis("modality", axis=0) \
            .reset_index(drop=False)
        return value_counts
        
    df_categorical = filtered_df[filtered_independent_var_names].loc[:, lambda df: df.dtypes == "O"]
    df_continuous = filtered_df[filtered_independent_var_names].loc[:, lambda df: df.dtypes != "O"]
    
    counts = [_count_modalities(var_name, serie) for var_name, serie in df_categorical.iteritems()]
    if df_categorical.shape[1] != 0:
        categorical_statistics = pd.concat(counts, ignore_index=True).assign(statistic="count_non_null")
    else:
        categorical_statistics = pd.DataFrame()
    
    if df_continuous.shape[1] != 0:
        mean_median = df_continuous \
            .loc[:, lambda df: df.dtypes != "O"] \
            .agg([np.mean,
                 np.median,
                 np.std,
                 np.min,
                 np.max,
                 "count"]) \
            .rename({"np.mean": "mean",
                     "np.median": "median",
                     "np.std": "std",
                     "amin": "min",
                     "amax": "max",
                     "count": "count_non_null"}) \
            .rename_axis("statistic", axis=0) \
            .reset_index(drop=False) \
            .melt(id_vars=["statistic"],
                  var_name="var_name",
                  value_name="value") \
            .assign(var_type="continuous")
    else:
        mean_median = pd.DataFrame()
    
    descriptive_statistics_df = pd.concat(
        [categorical_statistics, mean_median],
        ignore_index=True
    )
    return descriptive_statistics_df
