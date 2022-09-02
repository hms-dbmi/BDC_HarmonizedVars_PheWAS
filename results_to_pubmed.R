library(purrr)
library(tictoc)
library(tidyr)
library(dplyr)
library(readr)
library(stringr)
library(pubmedR)
library(bibliometrix)

# Testing PubMed API ------------------------------------------------------

api_key <- "9959092e61737e4f2f804a31fdaba6f46708"

# word <- "MAIC"
# query <- paste0(word, "*[Title/Abstract] AND english[LA] AND Journal Article[PT] AND 2017:2022[DP]")
#
# res <- pmQueryTotalCount(query = query, api_key = api_key)
#
# res$total_count
#
# D <- pmApiRequest(query = query, limit = res$total_count, api_key = api_key)
#
# data <- pubmedR::pmApi2df(D) %>% as_tibble()
#
# M <- bibliometrix::convert2df(D, dbsource = "pubmed", format = "api")
#
# results <- biblioAnalysis(M)
# summary(results)



# Reading phewas results --------------------------------------------------

continuous_results <- read_csv("results/dedup_continuous_significant.csv")

head(continuous_results)

## string treatment
variable_name_processing <- function(x) {
  x_wo_nonword_char <- str_replace_all(string = x,
                                       pattern = "[\\W\\d]+",
                                       replacement = " ")
  str_squish(x_wo_nonword_char) %>%
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
#TODO: execute that code
    print(phs)

    explanatory_variables <- df_queries$explanatory_variable[df_queries$phs_num == phs] %>%
      sample(x = ., size = ceiling(length(.)/5))

    harmonized_variables <- df_queries$harmonized_variable[df_queries$phs_num == phs] %>%
      sample(x = ., size = ceiling(length(.)/5))

    list_queries <- paste0("(",
                           explanatory_variables,
                           ")",
                           " AND ",
                           "(",
                           harmonized_variables,
                           ")",
                           " AND english[LA] AND Journal Article[PT]")
    length(list_queries)

    tic()
    my_pmQueryTotalCount <- function(list_queries) {
      indices_batch_queries <- 1:length(list_queries) %/% 10
      list_results_batch_queries <- vector(mode = "list", length = length(unique(indices_batch_queries)))
      for (n in unique(indices_batch_queries)) {
        print(n)
        batch_queries <- list_queries[indices_batch_queries == n]
        list_results_batch_queries[[n + 1]] <- lapply(batch_queries,
                                                      pmQueryTotalCount,
                                                      api_key = api_key)
        print("sleeping")
        Sys.sleep(10)
      }
      count_articles <- map(list_results_batch_queries, ~ map(.x, pluck, "total_count")) %>%
        unlist()
      return(count_articles)
    }

    count_articles <- my_pmQueryTotalCount(list_queries)
    toc()
    count_queries_df <- tibble(explanatory_variable = explanatory_variables,
                               harmonized_variables = harmonized_variables,
                               articles_count = count_articles)

    pubmed_queries_df <- df_queries %>%
      filter(phs_num == phs) %>%
      bind_cols(count_queries_df[, "articles_count"]) %>%
      arrange(desc(articles_count))
    write_csv(pubmed_queries_df, paste0("results/dedup_continuous_pubmed_", phs, ".csv"))

    timestamp()
    print("sleeping")
    # Sys.sleep(3600*)
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




# put results together ----------------------------------------------------

pubmed_file_names <- list.files("results", pattern = "pubmed", full.names = TRUE)

file_names <- lapply(pubmed_file_names, function(x) read_csv(x))

df_results_pubmed <- bind_rows(file_names) %>%
  arrange(desc(articles_count))


test <- left_join(continuous_results_dm,
                  df_results_pubmed[, c("explanatory_variable", "harmonized_variable", "Study", "articles_count")],
                  by =  c("explanatory_variable", "harmonized_variable", "Study")
) %>%
  arrange(desc(articles_count))
dim(continuous_results_dm)

test %>%
  distinct() %>%
  write_csv(file = "results/articles_count_pubmed.csv")


