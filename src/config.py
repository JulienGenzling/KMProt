import os
from dataclasses import dataclass
@dataclass
class Config:
    base_dir = os.path.expanduser(".")
    kernel_dir = os.path.join(base_dir, "./kernels") # kernels are heavy
    experiments_dir = os.path.join(base_dir, "experiments") # store json files from crossvalid.py / gridsearch.py
    ensemble_experiments_dir = os.path.join(base_dir, "ensemble_experiments") # store json files from ensemble_search.py