UPSTREAM = False
PICSURE_NETWORK_URL = "https://picsure.biodatacatalyst.nhlbi.nih.gov/picsure"
RESOURCE_ID = "02e23f52-f354-4e8b-992c-d37c8b9ba140"
token_file = "env_variables/token.txt"
RESULTS_PATH = "/run/user/1000/results_phewas"
with open(token_file, "r") as f:
    TOKEN = f.read()
    
batch_size = 500
