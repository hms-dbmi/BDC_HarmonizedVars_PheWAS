library(tidyr)
library(dplyr)
library(readr)
library(ggplot2)

continuous_results <- read_csv("results/dedup_continuous_significant.csv")
genoa <- continuous_results %>%
  filter(Study == "Genetic Epidemiology Network of Arteriopathy (GENOA) ( phs001238 )") %>%
  mutate(`Variable Effect` = abs(`Variable Effect`))


ggplot(genoa,
       aes(
         x = `Explanatory Variable`,
         y = `Explained Variable (harmonized)`,
         fill = `Variable Effect`
       )
) +
  geom_tile() +
  scale_fill_continuous(trans = "log10") +
  scale_x_discrete(labels = NULL, breaks = NULL) +
  scale_y_discrete(expand = expansion(0))

if (!dir.exists("results/heatmaps")) dir.create("results/heatmap")

for (study in unique(continuous_results$Study)) {
  title = study
  heatmap_association <- continuous_results %>%
    filter(Study == study) %>%
    mutate(`Variable Effect` = abs(`Variable Effect`)) %>%
  ggplot(aes(x = `Explanatory Variable`,
               y = `Explained Variable (harmonized)`,
               fill = `Variable Effect`)) +
    geom_tile() +
    scale_fill_continuous(trans = "log10") +
    scale_x_discrete(labels = NULL, breaks = NULL) +
    scale_y_discrete(expand = expansion(0)) +
  labs(title = title,
       y = "Harmonized variables",
       subtitle = "Continuous variables only, with p-value < 10.e-8") +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(face = "italic")) +
    theme_bw()
  print(heatmap_association)
  ggsave(plot = heatmap_association, filename = file.path("results/heatmap", paste0(title, ".png")))
}
