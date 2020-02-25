from typing import Union, List

import PicSureHpdsLib
import PicSureClient


def get_HPDS_connection(my_token: str,
                        PICSURE_network_URL: str = "https://picsure.biodatacatalyst.nhlbi.nih.gov/picsure",
                        resource_id: str = "02e23f52-f354-4e8b-992c-d37c8b9ba140"):
    client = PicSureClient.Client()
    connection = client.connect(PICSURE_network_URL, my_token)
    adapter = PicSureHpdsLib.Adapter(connection)
    db_con = adapter.useResource(resource_id)
    return db_con


def query_runner(resource: PicSureHpdsLib.Adapter.useResource,
                 to_select: Union[List[str], str] = None,
                 to_require: Union[List[str], str] = None,
                 to_anyof: Union[List[str], str] = None,
                 to_filter: dict = None,
                 result_type: str = "DataFrame",
                 **kwargs):
    
    def _testing_filter_parameter(to_filter):
        assert(isinstance(to_filter, dict)), "to_filter argument must be a dictionary; {0} passed instead"\
            .format(type(to_filter))
        for variable, filters in to_filter.items():
            assert (isinstance(filters, (str, int, float, list, dict))),\
                "type of filters passed for {0} variable is not supported".format(variable)
            if isinstance(filters, dict):
                assert(len(filters.keys()) != 0), "Empty dictionary provided to filter the following variable {0}"\
                    .format(variable)
                accepted_keywords = ["min", "max"]
                for sub_key in filters.keys():
                    if sub_key not in accepted_keywords:
                        raise ValueError("to_filter accepted keywords are {0}; '{1}' passed instead".format(accepted_keywords, sub_key))
    
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
            return query.getResultsDataFrame(**kwargs)
        elif result_type == "count":
            return query.getCount(**kwargs)
        else:
            raise ValueError("""
            {result_type} provided is not a recognized result_type for query object,
            instead should be one of the following: ["DataFrame", "count"]
            """.format(result_type=result_type))

    if to_filter is not None:
        _testing_filter_parameter(to_filter)
    query = _build_query(resource, to_select, to_filter, to_require, to_anyof)
    return _run_query(query, result_type, **kwargs)


if __name__ == '__main__':
    import os
    import pandas as pd
    import numpy as np

    from NHLBI_BioData_Catalyst.python.python_lib.utils import get_multiIndex_variablesDict
    
    PICSURE_network_URL = "https://biodatacatalyst.integration.hms.harvard.edu/picsure"
    resource_id = "02e23f52-f354-4e8b-992c-d37c8b9ba140"
    token_file = "token.txt"
    with open(token_file, "r") as f:
        my_token = f.read()
    client = PicSureClient.Client()
    connection = client.connect(PICSURE_network_URL, my_token)
    adapter = PicSureHpdsLib.Adapter(connection)
    resource = adapter.useResource(resource_id)

    # resource = get_HPDS_connection(PICSURE_network_URL, resource_id, my_token)
    plain_variablesDict = resource.dictionary().find("COPDGene").DataFrame()
    variablesDict = get_multiIndex_variablesDict(plain_variablesDict)
    mask = variablesDict["simplified_name"] == "How old were you when you completely stopped smoking? [Years old]"
    yo_stop_smoking_varname = variablesDict.loc[mask, "name"].values[0]
    # %%
    mask_cat = variablesDict["categorical"] == True
    mask_count = variablesDict["observationCount"]
    varnames = variablesDict.loc[mask_cat & mask_count, "name"]

    resource = get_HPDS_connection(token, PICSURE_network_URL, resource_id)
    df = query_runner(resource,
                      to_select=varnames,
                      to_require=varnames[2],
                      to_filter={yo_stop_smoking_varname: {"min": 20, "max": 70}},
                      result_type="DataFrame",
                      timeout=30,
                      low_memory=False
                      )

