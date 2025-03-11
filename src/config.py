from dataclasses import dataclass
@dataclass
class Config:
    kernel_dir = "/Data/kmprotdata" # kernels are heavy
    experiments_dir = "experiments2" # store json files from crossvalid.py / gridsearch.py
    ensemble_experiments_dir = "ensemble_experiments" # store json files from ensemble_search.py