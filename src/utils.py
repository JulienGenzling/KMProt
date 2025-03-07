import os
import json

from src.kernel import *
from src.fitter import *

def write_results(dataset, fitter, kernel, acc, output_file):
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
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            experiments = json.load(f)
    else:
        experiments = []

    experiments.append(new_experiment)

    with open(output_file, "w") as f:
        json.dump(experiments, f, indent=4)

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