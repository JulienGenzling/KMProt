import os
import json

from src.kernel import *
from src.fitter import *


def write_results(dataset, fitter, kernel, acc, output_folder="experiments", weight=False):
    os.makedirs(output_folder, exist_ok=True)

    existing_files = [
        f
        for f in os.listdir(output_folder)
    ]
    num_files = len(existing_files)

    if weight:
        new_file_name = f"ensemble_experiment_{num_files + 1}.json"
    else:
        new_file_name = f"experiment_{num_files + 1}.json"
    new_file_path = os.path.join(output_folder, new_file_name)

    kernel_params = kernel.kernel_params_list if weight else kernel.params
    new_experiment = {
        "dataset": dataset.k,
        "fitter": fitter.params,
        "kernel": kernel_params,
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


if __name__ == "__main__":
    from src.dataset import Dataset
    dataset = Dataset(1)
    kernel_params = {'name': 'mismatch', 'k': 9, 'm': 1}
    fitter_params = {'name': 'svm', 'C': 1, 'tol': 1e-4}
    verbose = True
    kernel, fitter = get_obj(dataset, kernel_params, fitter_params, verbose)