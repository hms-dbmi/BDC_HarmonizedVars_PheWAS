# Launch App

# Install/load packages
paket <- function(pak){
  not_installed <- pak[!(pak %in% rownames(installed.packages()))]
  if (length(not_installed))
    install.packages(not_installed, dependencies = TRUE,repos='http://cran.us.r-project.org')
  sapply(pak, library, character.only = TRUE)
}
install = c("shiny", "shinydashboard", "shinyWidgets", "esquisse", "dplyr", "DT", "stringr", "tidyr")
paket(install)

whole_data <- data.table::fread(file = "../python/results/results_formated.csv",
                                sep=",",
                                header = TRUE,
                                na.strings = c("NA", ""))
whole_data <- whole_data[!is.na(whole_data[["pvalue"]])]
whole_data[["names_wout_backslashes"]] <- whole_data[["Independent Variable Complete Name"]]
study_names <- whole_data[["BDC study"]] %>% unique()
not_all_na <- function(x) {!all(is.na(x))}

# Launching the ShinyApp

source("app.R")
runApp(app)
