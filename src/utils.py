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
    
def get_fitter(fitter_params):
    if fitter_params["name"] == "svm":
        fitter = SVM(**fitter_params)
    elif fitter_params["name"] == "klr":
        fitter = KLR(**fitter_params)
    elif fitter_params["name"] == "krr":
        fitter = KRR(**fitter_params)
    else:
        NotImplementedError("Fitter not implemented")
    return fitter

def get_obj(dataset, kernel_params, fitter_params, verbose):
    if kernel_params["name"] == "spectrum":
        kernel = MultiSpectrumKernel(dataset, **kernel_params, verbose=verbose)
    if kernel_params["name"] == "mismatch":
        kernel = MismatchKernel(dataset, **kernel_params, verbose=verbose)
    else:
        NotImplementedError("Kernel not implemented")
    fitter = get_fitter(fitter_params)
    return kernel, fitter

def get_obj_ensemble(dataset, kernel_configs, fitter_params, verbose):
    kernel = WeightedSumKernel(dataset, kernel_configs, verbose=verbose)
    fitter = get_fitter(fitter_params)
    return kernel, fitter


if __name__ == "__main__":
    from src.dataset import Dataset
    dataset = Dataset(1)
    kernel_params = {'name': 'mismatch', 'k': 9, 'm': 1}
    fitter_params = {'name': 'svm', 'C': 1, 'tol': 1e-4}
    verbose = True
    kernel, fitter = get_obj(dataset, kernel_params, fitter_params, verbose)