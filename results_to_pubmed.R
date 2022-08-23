library(tidyr)
library(dplyr)
library(readr)
library(stringr)
library(pubmedR)
library(bibliometrix)

# Testing PubMed API ------------------------------------------------------

api_key <- "9959092e61737e4f2f804a31fdaba6f46708"

word <- "MAIC"
query <- paste0(word, "*[Title/Abstract] AND english[LA] AND Journal Article[PT] AND 2017:2022[DP]")

res <- pmQueryTotalCount(query = query, api_key = api_key)

res$total_count

D <- pmApiRequest(query = query, limit = res$total_count, api_key = api_key)

data <- pubmedR::pmApi2df(D) %>% as_tibble()

M <- bibliometrix::convert2df(D, dbsource = "pubmed", format = "api")

results <- biblioAnalysis(M)
summary(results)



# Reading phewas results --------------------------------------------------

continuous_results <- read_csv("results/dedup_continuous_significant.csv")

head(continuous_results)

## string treatment
variable_name_processing <- function(x) {
  x_wo_starting_numbers <- sub(pattern = "^[0-9]+[\\W]*",
                               replacement = "",
                               perl = TRUE,
                               x = x)
  str_squish(x_wo_starting_numbers) %>%
    sub(pattern = "'", replacement = "", x = ., fixed = TRUE) %>%
    str_to_lower()
}

continuous_results_dm <- continuous_results %>%
  rename(
    "explanatory_variable" = `Explanatory Variable`,
    "harmonized_variable" = `Explained Variable (harmonized)`
  ) %>%
  mutate(explanatory_variable = variable_name_processing(explanatory_variable))

df_queries <- continuous_results_dm[, c("explanatory_variable", "harmonized_variable", "Study")] %>%
  distinct()

query_w_harmonized <- TRUE
df_queries$phs_num <- stringr::str_extract(df_queries$Study, pattern = "phs[0-9]{6}")

list_phs <- table(df_queries$phs_num) %>%
  as.data.frame() %>%
  as_tibble() %>%
  rename("phs_num" = "Var1") %>%
  arrange(Freq) %>%
  pull(phs_num) %>%
  as.character()

if (query_w_harmonized) {
  for (phs in list_phs) {
    explanatory_variables <- df_queries$explanatory_variable[df_queries$phs_num == phs]
    harmonized_variables <- df_queries$harmonized_variable[df_queries$phs_num == phs]

    list_queries <- paste0("(",
                           explanatory_variables,
                           "[Title/Abstract])",
                           " AND ",
                           "(",
                           harmonized_variables,
                           "[Title/Abstract])",
                           " AND english[LA] AND Journal Article[PT] AND 2017:2022[DP]")
    length(list_queries)

    tic()
    n_queries <- lapply(list_queries,
                        pmQueryTotalCount,
                        api_key = api_key)
    time <- toc()
    count_queries_df <- tibble(explanatory_variable = explanatory_variables,
                               harmonized_variables = harmonized_variables,
                               articles_count = sapply(n_queries, function(x) x[["total_count"]]))

    pubmed_queries_df <- df_queries %>%
      filter(phs_num == phs) %>%
      bind_cols(count_queries_df[, "articles_count"]) %>%
      arrange(desc(articles_count))

    Sys.sleep(3600)
    write_csv(pubmed_queries_df, paste0("results/dedup_continuous_pubmed_", phs, ".csv"))
  }
} else {
  query_explanatory <- "*[Title/Abstract])"
  list_queries <- paste0("(",
                         unique(df_queries$explanatory_variable),
                         "[Title/Abstract]) AND english[LA] AND Journal Article[PT] AND 2017:2022[DP]")
  length(list_queries)
  tic()
  n_queries <- lapply(list_queries,
                      pmQueryTotalCount,
                      api_key = api_key)
  time <- toc()
}


length(n_queries)

count_queries_df <- tibble(explanatory_variable = unique(continuous_results_dm$explanatory_variable),
                           articles_count = sapply(n_queries, function(x) x[["total_count"]]))


pubmed_queries_df <- left_join(continuous_results_dm,
                               count_queries_df,
                               by = "explanatory_variable") %>%
  arrange(desc(articles_count))


View(pubmed_queries_df)

write_csv(pubmed_queries_df, "results/dedup_continuous_pubmed.csv")

# Eventually map the list of strings to MESH terms


