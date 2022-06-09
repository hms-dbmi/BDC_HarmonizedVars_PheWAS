library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(yaml)
library(purrr)
library(data.table)

parameters_exp = read_yaml("env_variables/parameters_exp.yaml")

date_results = "220426_202807"
path_results = file.path(parameters_exp$results_path, date_results)
path_monitor = file.path(path_results, "monitor_process.csv")

dict_harmonized_variables = read_csv("env_variables/df_harmonized_variables.csv",
                                     col_types = "cccccc") %>%
as.data.table()

multiindex_variablesDict = read_csv("env_variables/multiIndex_variablesDict.csv",
                                    col_types = "cccccccdlcdddccli") %>% as.data.table()
dict_independent_variables = read_csv("env_variables/df_eligible_variables.csv", col_types = "ccicc") %>%
                               left_join(multiindex_variablesDict[, c("name", "simplified_name")], 
                                    by=c("var_name" = "name")) %>%
as.data.table()
now = Sys.time()
print(now)

map_progress <- function(.x, .f, ...) {
  .f <- purrr::as_mapper(.f, ...)
  pb <- progress::progress_bar$new(total = length(.x), force = TRUE)
  
  f <- function(...) {
    pb$tick()
    .f(...)
  }
  purrr::map(.x, f, ...)
}

L <- map_progress(.f = fread,
         .x = list.files(file.path(path_results, "association_statistics"), full.name = TRUE, recursive = TRUE),
         colClasses=list(
             "character" = "dependent_var_id", 
             "character" = "independent_var_id", 
             "character" = "dependent_var_modality",
             "character" = "ref_modality_dependent",
             "character" = "independent_var_modality",
             "character" = "ref_modality_independent",
             "character" = "indicator", 
             "double" = "value"), 
         data.table = TRUE, 
         header = TRUE
         )
then = Sys.time()
print(then)
then - now
association_table = bind_rows(L)

n_tests_ran = association_table[, c("dependent_var_id", "independent_var_id")] %>%
    distinct() %>%
    nrow()
print(n_tests_ran)
print(n_tests_significant)

pvalues_name = c("pvalue_LRT_LR", "pvalue_LRT_LogR")
significant_associations = association_table[indicator %in% pvalues_name, ] %>%
                 .[value < 0.05/n_tests_ran, list(dependent_var_id, independent_var_id, 
                                                 value, indicator, dependent_var_modality, 
                                                 independent_var_modality)] 
setnames(significant_associations, "value", "significance")

significant_association_table = association_table[significant_associations, on = .(dependent_var_id, independent_var_id)]
#setnames(significant_association_table, "i.value", "pvalue")

wide_significant_table <- significant_association_table %>% 
  unique(by = c("dependent_var_id", "independent_var_id", "indicator", "significance")) %>%
  dcast(formula = dependent_var_id + independent_var_id + significance ~ indicator, value.var = "value") %>%
  unique() %>%
  .[dict_harmonized_variables[, list(
      harmonized_var_name = renaming_variables_nice,
      dependent_var_id,
      harmonized_var_type = var_type)], on = .(dependent_var_id)] %>%
  .[dict_independent_variables[, list(independent_var_name = var_name,
                                      study,
                                      phs, 
                                      independent_var_id,
                                      independent_simplified_name = simplified_name)], 
   on = .(independent_var_id), 
   nomatch = NULL]

categorical_significant =  wide_significant_table[harmonized_var_type %in% c("binary", "categorical"), ] %>%
  as_tibble() %>%
  select(where(function(x) !all(is.na(x)))) %>%
  mutate(OR = round(exp(coeff_LogR), 2), 
         CI_OR = paste0(round(exp(coeff_LogR_lb), 2), " ; ", round(exp(coeff_LogR_ub), 2))) %>%
  rename("Explanatory Variable" = "independent_simplified_name", 
        "Explanatory Variable (complete name)" = "independent_var_name", 
        "Explained Variable (harmonized)" = "harmonized_var_name", 
        "Sample Size" = "nonNA_crosscount", 
        "Study" = "study") %>%
  select(all_of(c("Explanatory Variable", "Explained Variable (harmonized)", "OR", "CI_OR", "significance", "Sample Size", "Explanatory Variable (complete name)", "Study"))) %>%
  arrange(significance, desc(OR))
               

continuous_significant = wide_significant_table[harmonized_var_type %in% c("continuous"), ] %>% 
  as_tibble() %>%
  select(where(function(x) !all(is.na(x)))) %>%
  mutate(`Variable Effect` = round(coeff_LR, 2), 
         `CI Variable Effect` = paste0(round(coeff_LR_lb, 2), " ; ", round(coeff_LR_ub, 2))) %>%
  rename("Explanatory Variable" = "independent_simplified_name", 
        "Explanatory Variable (complete name)" = "independent_var_name", 
        "Explained Variable (harmonized)" = "harmonized_var_name", 
        "Sample Size" = "nonNA_crosscount", 
        "Study" = "study") %>%
  select(all_of(c("Explanatory Variable", "Explained Variable (harmonized)", "Variable Effect", "CI Variable Effect", "significance", "Sample Size", "Explanatory Variable (complete name)", "Study"))) %>%
  arrange(significance, desc(`Variable Effect`))
               
save(wide_significant_table, file = file.path(path_results, "wide_significant_table.RData"))
save(association_table, file = file.path(path_results, file = "association_table.RData"))
categorical_significant %>% write_csv(file.path(path_results, "categorical_significant.csv"))
continuous_significant %>% write_csv(file.path(path_results, "continuous_significant.csv"))


# 
# sign_association_table = association_table[significant_associations[, .(significance)],
#                                            on = c("dependent_var_id", "independent_var_id"),
#                                            nomatch = NULL]
# sign_association_table.pivot_wider()


