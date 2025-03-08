import os
import json

from src.kernel import *
from src.fitter import *

class Experiment:
    def __init__(self, exp_dict_or_file, experiments_dir="."):
        if isinstance(exp_dict_or_file, dict):
            self.experiment = exp_dict_or_file
        elif isinstance(exp_dict_or_file, str):
            file_path = os.path.join(experiments_dir, exp_dict_or_file)
            with open(file_path, "r") as f:
                self.experiment = json.load(f)
        else:
            raise ValueError("Invalid input: expected a dictionary or a filename")

    def __eq__(self, other):
        if not isinstance(other, Experiment):
            return False
        return self.experiment == other.experiment

    def __repr__(self):
        return f"Experiment({self.experiment})"

def write_results(dataset, fitter, kernel, acc, output_folder='experiments'):
    os.makedirs(output_folder, exist_ok=True)

    existing_files = [f for f in os.listdir(output_folder) if f.startswith('experiment_') and f.endswith('.json')]
    num_files = len(existing_files)

    new_file_name = f'experiment_{num_files + 1}.json'
    new_file_path = os.path.join(output_folder, new_file_name)

    new_experiment = {
        "dataset": dataset.k,
        "fitter": {
            "name": fitter.__class__.__name__,
            "params": fitter.params,
        },
        "kernel": {
            "name": kernel.__class__.__name__,
            "params": kernel.params,
        },
        "results": acc,
    }

    with open(new_file_path, "w") as f:
        json.dump(new_experiment, f, indent=4)


def get_obj(dataset, kernel_params, fitter_params, verbose):
    if kernel_params["name"] == "spectrum":
        kernel = MultiSpectrumKernel(dataset, **kernel_params, verbose=verbose)
    if kernel_params["name"] == "mismatch":
        kernel = MismatchKernel(dataset, **kernel_params, verbose=verbose)
    else:
        NotImplementedError("Kernel not implemented")
    if fitter_params["name"] == "svm":
        fitter = SVM(**fitter_params)
    elif fitter_params["name"] == "klr":
        fitter = KLR(**fitter_params)
    elif fitter_params["name"] == "krr":
        fitter = KRR(**fitter_params)
    else:
        NotImplementedError("Fitter not implemented")
    return kernel, fitter


def find_best_params(dataset):

    with open("experiments.json", "r") as file:
        experiments = json.load(file)

    sorted_experiments = sorted(experiments, key=lambda x: x["results"], reverse=True)
    best_2 = sorted_experiments[:2]
    for i, exp in enumerate(best_2, start=1):
        print(f"Best {i}:")
        print(f"  Dataset: {exp['dataset']}")
        print(f"  Fitter: {exp['fitter']}")
        print(f"  Kernel: {exp['kernel']}")
        print(f"  Results: {exp['results']}\n")
    pass