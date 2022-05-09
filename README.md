# PheWAS using PICSURE API

Goal of the project is to perform a PheWAS analysis using the PICSURE API, across the different TOPMed and TOPMed related studies from BioDataCatalyst, by using any of the Harmonized Phenotypes Variables. 


## Repo organization

Files: 

- `setup_environment.py`: run that script first, before every pipeline run: creates the environment variables, parameters, list of variables to query, batch groups, etc ...
- `run_PheWAS.py`: main script, run a phewas analysis for one single batch of variables
- `span_run_PheWAS.sh`: span the run_PheWAS script into multiple processes
- `compile_run_PheWAS.py`: merge individual results output of multiple run_PheWAS.py 
- `count_n_subject.py`: temporary script, not part of the main pipeline, only useful to count the total number of subjects for a single study
- 

Folders:

- `python_lib` : 
    - `utils.py` contains mainly functions used to enhance PICSURE API, as get_multiIndex_VariablesDict
    - `wrappers.py` contains user defined higher-level functions to ease retrieving patient-level data using PICSURE API
    - `PheWAS_funcs.py` wrappers functions for one phewas analysis: used to scale up the phewas across multiple independent studies
    - `descriptive_scripts`: script used to describe each individual studies at a high level, ie number of categorical, continuous variables, ratio of null/non null values among one study variables
- `studies_stats`: contain results for the high level statistics for each study
- `env_variables`: information to select studies to be used

