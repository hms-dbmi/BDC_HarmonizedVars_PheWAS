# Title     : Wrappers HPDS library
# Objective :
# Created by: Arnaud
# Created on: 2020-02-04

### DOC 
# How to create named list for to_filter to work properly: to_filter = setNames(list(phs_copdgene), list(consent_dic[["name"]]))
#
#
get_HPDS_connection <- function(PICSURE_network_URL,
                                resource_id,
                                my_token) {
    myconnection <- picsure::connect(url = PICSURE_network_URL, token = my_token)
    hpds::get.resource(myconnection, resourceUUID = resource_id)
}

query_runner <- function(resource,
                         to_select = NULL,
                         to_filter = NULL,
                         to_require = NULL,
                         to_anyof = NULL,
                         result_type = "dataframe",
                         ...) {
    .test_filter_parameter <- function(to_filter) {
        if (!is.list(to_filter)) {
                stop(paste("filter parameter should be a list",
                            class(to_filter),
                            "passed instead"))
        } else if (is.null(names(to_filter))) {
                stop("filter parameter should be a named list,
                but a non-named list has been passed instead")
        }
        for (num_var in length(to_filter)) {
            variable_filter_values = to_filter[num_var]
            if (! typeof(variable_filter_values) %in% c("double", "character", "list")) {
                stop(paste("type of list individual values for 'to_filter' parameter
                should be either 'double', 'character' or 'list', but",
                typeof(variable_filter_values), "provided instead"))
            } else if (typeof(variable_filter_values) == "list") {
               if (length(variable_filter_values) == 0) {
                stop(paste("empty filter values provided for variable",
                     names(variable_filter_values)))
                } else if (!is.null(names(to_filter[[num_var]]))) {
                    accepted_keywords = list("min", "max", "value")
                    for (sub_key in names(to_filter[[num_var]])) {
                        if (! sub_key %in% accepted_keywords) {
                            stop(paste("to_filter accepted keywords are ",
                            accepted_keywords, "; ", sub_key, "passed instead."))
                        }
                    }
                }
            }
        }
    }

    .build_query <- function(query,
                         to_select,
                         to_filter,
                         to_require,
                         to_anyof) {
        if (! is.null(to_select)) hpds::query.select.add(query = query, keys = to_select)
        if (! is.null(to_anyof)) hpds::query.anyof.add(query = query, keys = to_anyof)
        if (! is.null(to_require)) hpds::query.require.add(query = query, keys = to_require)
        if (! is.null(to_filter)) {
            for (variable in names(to_filter)) {
                query_args = to_filter[[variable]]
                query_args[["query"]] = query
                query_args[["keys"]] = variable
                do.call(hpds::query.filter.add, query_args)
            }
        }
        return(query)
    }

    query = hpds::new.query(resource = resource)
    if (! is.null(to_filter)) .test_filter_parameter(to_filter)
    query <- .build_query(query,
                          to_select = to_select,
                          to_filter = to_filter,
                          to_require = to_require,
                          to_anyof = to_anyof)
    hpds::query.run(query = query, result.type = result_type)
}
