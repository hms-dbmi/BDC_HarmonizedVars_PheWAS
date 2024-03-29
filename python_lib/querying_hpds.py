from typing import Union, List
import contextlib
import io
import re

import pandas as pd
import numpy as np

import PicSureHpdsLib
import PicSureBdcAdapter
import PicSureClient

from env_variables.env_variables import TOKEN, RESOURCE_ID, PICSURE_NETWORK_URL
from python_lib.utils import get_multiIndex_variablesDict


class HpdsHTTPError(Exception):
    """
    Raised to catch HTTP error messages from PICSURE
    """


class EmptyDataFrameError(Exception):
    """
    Raised to catch Data Frame returned empty
    """

    
def raise_EmptyDataFrameError(function):
    def error_handling_function(*args, **kwargs):
        df_output = function(*args, **kwargs)
        if df_output is None:
            raise EmptyDataFrameError("No valid column names selected, output DF is empty")
        else:
            return df_output
    return error_handling_function


def raise_HpdsError(function):
    def errors_handling_function(*args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()) as f:
            output = function(*args, **kwargs)
        output_message = f.getvalue()
        print(output_message)
        if re.match("ERROR: HTTP response was bad", output_message) is not None:
            raise HpdsHTTPError(output_message)
        else:
            return output
    return errors_handling_function

def retry_query(function):
    def retry_query_function(*args, **kwargs):
        n = 0
        def _snippet(*args, **kwargs):
            with contextlib.redirect_stdout(io.StringIO()) as f:
                output = function(*args, **kwargs)
            output_message = f.getvalue()
            return output, output_message
        output, output_message = _snippet(*args, **kwargs)
        while re.match("ERROR: HTTP response was bad", output_message) is not None:
            n += 1
            import time
            time.sleep(120)
            if n <= 3:
                output, output_message = _snippet(*args, **kwargs)
            else:
                return output_message
        else:
            return output
    return retry_query_function


@raise_HpdsError
def get_HPDS_connection(my_token: str = TOKEN,
                        URL_data: str = PICSURE_NETWORK_URL,
                        resource_id: str = RESOURCE_ID):
    if my_token is None:
        with open("env_variables/token.txt", "r") as f:
            my_token = f.read()
    print(my_token)
    print(URL_data)
    print(resource_id)
    client = PicSureClient.Client()
    connection = client.connect(URL_data, my_token)
    
    if re.search("biodatacatalyst", URL_data) is not None:
        adapter = PicSureBdcAdapter.Adapter(connection)
    else:
        adapter = PicSureHpdsLib.Adapter(connection)
    db_con = adapter.useResource(resource_id)
    return db_con


@raise_EmptyDataFrameError
@raise_HpdsError
def query_runner(resource: PicSureHpdsLib.Adapter.useResource,
                 to_select: Union[List[str], str] = None,
                 to_require: Union[List[str], str] = None,
                 to_anyof: Union[List[str], str] = None,
                 to_filter: dict = None,
                 result_type: str = "DataFrame",
                 **kwargs):
    def _testing_filter_parameter(to_filter):
        assert (isinstance(to_filter, dict)), "to_filter argument must be a dictionary; {0} passed instead" \
            .format(type(to_filter))
        for variable, filters in to_filter.items():
            assert (isinstance(filters, (str, int, float, list, dict))), \
                "type of filters passed for {0} variable is not supported".format(variable)
            if isinstance(filters, dict):
                assert (len(filters.keys()) != 0), "Empty dictionary provided to filter the following variable {0}" \
                    .format(variable)
                accepted_keywords = ["min", "max"]
                for sub_key in filters.keys():
                    if sub_key not in accepted_keywords:
                        raise ValueError(
                            "to_filter accepted keywords are {0}; '{1}' passed instead".format(accepted_keywords,
                                                                                               sub_key))
    
    def _build_query(resource,
                     to_select,
                     to_filter,
                     to_require,
                     to_anyof):
        query = resource.query()
        if to_select is not None:
            query.select().add(to_select)
        if to_require is not None:
            query.require().add(to_require)
        if to_anyof is not None:
            query.anyof().add(to_anyof)
        if to_filter is not None:
            for variable, filters in to_filter.items():
                if isinstance(filters, dict):
                    query.filter().add(variable, **filters)
                else:
                    query.filter().add(variable, filters)
        return query
    
    def _run_query(query, result_type, **kwargs):
        if result_type == "DataFrame":
            return query.getResultsDataFrame(keep_default_na=True,
                                             na_values=["Null", "NAN"],
                                             **kwargs)
        elif result_type == "count":
            return query.getCount(**kwargs)
        elif result_type == "string":
            return query.getResults(**kwargs)
        else:
            raise ValueError("""
            {result_type} provided is not a recognized result_type for query object,
            instead should be one of the following: ["DataFrame", "count", "string"]
            """.format(result_type=result_type))
    
    if to_filter is not None:
        _testing_filter_parameter(to_filter)
    query = _build_query(resource, to_select, to_filter, to_require, to_anyof)
    return _run_query(query, result_type, **kwargs)


@raise_EmptyDataFrameError
@raise_HpdsError
def get_mock_df(resource=None):
    if resource is None:
        with open("env_variables/token.txt", "r") as f:
            token = f.read()
        resource = get_HPDS_connection(token)
    study_name = "Genetic Epidemiology of COPD (COPDGene) Funded by the National Heart, Lung, and Blood Institute ( phs000179 )"
    phs_list = ['phs000179.c0', 'phs000179.c1']
    plain_variablesDict = resource.dictionary().find("COPDGene").DataFrame()
    variablesDict = get_multiIndex_variablesDict(plain_variablesDict)
    selected_var = variablesDict.loc[study_name, "name"].values.tolist()[:-1:10]
    facts = query_runner(resource=resource,
                         to_anyof=selected_var,
                         result_type="DataFrame")
    return facts


def get_batch_groups(variablesDict: pd.DataFrame,
                     batch_size,
                     assign: bool):
    # Return vector of integer, representing batch group indices, useful to process varibles in batches for the PheWAS pipeline
    len_dic = variablesDict.shape[0]
    batch_indices = []
    for batch_group in range(0, int(np.ceil(len_dic / batch_size))):
        batch_indice = 0
        while batch_indice < batch_size:
            batch_indices.append(batch_group)
            batch_indice += 1
    batch_indices = batch_indices[:len_dic]
    if assign is True:
        return variablesDict.assign(batch_group=batch_indices)
    else:
        return batch_indices

@raise_HpdsError
def get_whole_dic(resource=None,
                  batch_size: int = None,
                  write: bool = False):
    
    if resource is None:
        with open("token.txt", "r") as f:
            token = f.read()
        resource = get_HPDS_connection(token)
    
    plain_variablesDict = resource.dictionary().find().DataFrame()
    variablesDict = get_multiIndex_variablesDict(plain_variablesDict)
    
    if batch_size is not None:
        batch_indices = get_batch_groups(variablesDict, batch_size, False)
        variablesDict["batch_group"] = batch_indices
        with open("./env_variables/batch_list.txt", "w+") as f:
            for line in variablesDict["batch_group"].astype(str).unique().tolist():
                f.write("{0}\n".format(line))
    
    if write is True:
        variablesDict.to_csv("env_variables/multiIndex_variablesDict.csv")
        return
    else:
        return variablesDict

@raise_HpdsError
def get_studies_dictionary(resource=None) -> pd.DataFrame:
    """
    Return dataframe with names and phs number of the studies with at least one
    subject with at least one harmonized variable information
    :param resource:
    :return:
    """
    if resource is None:
        resource = get_HPDS_connection()
    
    whole_dic = get_whole_dic(resource, batch_size=None, write=False)
    studies_names = whole_dic.index.get_level_values(0).drop_duplicates().to_frame()
    studies_names["phs"] = studies_names.index.str.extract("(phs[0-9]+)", expand=False)
    studies_names = studies_names.loc[:, ["phs"]]
    
    harmonized_variables = resource.dictionary().find("harmonized").DataFrame().index.tolist()
    harmonized_df = query_runner(resource, to_anyof=harmonized_variables)
    harmonized_consents = harmonized_df["\\_harmonized_consent\\"] \
        .str.extract("(^\w*?(?=\.))", expand=False) \
        .drop_duplicates() \
        .to_frame() \
        .assign(harmonized=True) \
        .set_index("\\_harmonized_consent\\")
    
    studies_dictionary = studies_names.join(harmonized_consents, on="phs") \
                                      .fillna({"harmonized": False})\
                                      .rename({"level_0": "study_name"}, axis=1)\
                                      .rename_axis("study_name", axis=1)
    return studies_dictionary

@raise_EmptyDataFrameError
@raise_HpdsError
def get_one_study(resource,
                  phs: str,
                  studies_info: pd.DataFrame,
                  variablesDict: pd.DataFrame = None,
                  result_type: str = "DataFrame",
                  harmonized_vars: bool = True,
                  **kwargs) -> pd.DataFrame:
    if variablesDict is None:
        plain_variablesDict = resource.dictionary().find().DataFrame()
        variablesDict = get_multiIndex_variablesDict(plain_variablesDict)
    consent_var = '\\_Consents\\Short Study Accession with Consent Code\\'
    phs_list = studies_info.loc[phs, "phs_list"]
    study_name = studies_info.loc[phs, "BDC_study_name"]
    variablesDict = variablesDict.set_index("level_0")
    print(variablesDict.index.unique())
    studies_var = variablesDict.loc[study_name, "name"].values.tolist()
    if harmonized_vars is True:
        print(variablesDict.index.unique())
        harmonized_vars = variablesDict.loc["DCC Harmonized data set", "name"].values.tolist()
    else:
        harmonized_vars = None
    facts = query_runner(resource=resource,
                         to_select=harmonized_vars,
                         to_anyof=studies_var,
                         to_filter={consent_var: phs_list},
                         result_type=result_type,
                         **kwargs)
    return facts
