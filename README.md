# PheWAS using PICSURE API

Goal of the project is to perform a PheWAS analysis using the PICSURE API, across the different TOPMed and TOPMed related studies from BioDataCatalyst, by using any of the Harminized Phenotypes Variables. 


## Repo organization

Files: 

- run_PheWAS.py: main script, run a phewas analysis for one single study
- span_run_PheWAS.sh: span the run_PheWAS script into multiple processes
- studies_stats.py: run high level statistics for one study
- span_studies_stats.sh: span the run_studies_stats script into multi processes
- `compile_variables_stats.py`: merge individual study results of run_PheWAS.py 
- `compile_studies_stats.py`: merge individual study results of studies_stats.py

Folders:

- `python_lib` : 
    - `utils.py` contains mainly functions used to enhance PICSURE API, as get_multiIndex_VariablesDict
    - `wrappers.py` contains user defined higher-level functions to ease retrieving patient-level data using PICSURE API
    - `PheWAS_funcs.py` wrappers functions for one phewas analysis: used to scale up the phewas across multiple independent studies
    - `descriptive_scripts`: script used to describe each individual studies at a high level, ie number of categorical, continuous variables, ratio of null/non null values among one study variables
- `results`: contain results for the PheWAS analysis
- `studies_stats`: contain results for the high level statistics for each studizes
- `env_variables`: information to select studies to be used

