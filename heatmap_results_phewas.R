library(tidyr)
library(dplyr)
library(readr)
library(ggplot2)

continuous_results <- read_csv("results/dedup_continuous_significant.csv")
genoa <- continuous_results %>%
  filter(Study == "Genetic Epidemiology Network of Arteriopathy (GENOA) ( phs001238 )")

ggplot(genoa,
       aes(
         x = `Explanatory Variable`,
         y = `Explained Variable (harmonized)`,
         fill = `Variable Effect`
       )
) +
  geom_tile() +
  scale_x_discrete(labels = NULL, breaks = NULL)

if (!dir.exists("results/heatmaps")) dir.create("results/heatmap")

for (study in unique(continuous_results$Study)) {
  title = study
  heatmap_association <- continuous_results %>%
    filter(Study == study) %>%
    ggplot(aes(x = `Explanatory Variable`,
               y = `Explained Variable (harmonized)`,
               fill = `Variable Effect`)) +
    geom_tile() +
    scale_x_discrete(labels = NULL, breaks = NULL) +
    # scale_fill_continuous(high = "#075AFF",
    #                       low = "#FF0000") +
    labs(title = title,
         y = "Harmonized variables",
         subtitle = "Continuous variables only, with p-value < 10.e-8") +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(face = "italic"))
  print(heatmap_association)
  ggsave(plot = heatmap_association, filename = file.path("results/heatmap", paste0(title, ".png")))
}
