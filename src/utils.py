import os
import json

from src.kernel import *
from src.fitter import *
from src.config import Config


def write_results(dataset, fitter, kernel, acc, weight=False):
    if weight:
        output_folder = Config.ensemble_experiments_dir
    else:
        output_folder = Config.experiments_dir
        
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
    return fitter, kernel

def get_obj_ensemble(dataset, kernel_configs, fitter_params, verbose):
    kernel = WeightedSumKernel(dataset, kernel_configs, verbose=verbose)
    fitter = get_fitter(fitter_params)
    return fitter, kernel

def load_experiments(experiments_dir):
    existing_experiments = []
    if os.path.exists(experiments_dir):
        for filename in os.listdir(experiments_dir):
            if filename.endswith(".json"):
                with open(
                    os.path.join(experiments_dir, filename), "r"
                ) as f:
                    exp = json.load(f)
                existing_experiments.append(exp)
    return existing_experiments

def infer(kernel, fitter, sequences, ensemble, datafold):
    test_phis = [kernel._get_phi(seq) for seq in sequences]
    predictions = []

    if not ensemble:
        train_norms = np.array(
            [np.sqrt(kernel.dot(phi_tr, phi_tr)) for phi_tr in kernel.phis]
        )
        for phi in tqdm(test_phis, desc=f"Predicting fold {datafold}"):
            phi_norm = np.sqrt(kernel.dot(phi, phi))
            dots = np.array([kernel.dot(phi, phi_tr) for phi_tr in kernel.phis])
            Kx = dots / (train_norms * phi_norm)
            pred = (
                np.dot(fitter.alpha * fitter.sv_label, Kx[fitter.sv_indices])
                + fitter.intercept
            )
            predictions.append(np.sign(pred))
    else:
        for phi_test in tqdm(test_phis, desc=f"Predicting fold {datafold}"):
            Kx = np.zeros(next(iter(kernel.phis.values())).shape[0])
            for weight, phis in kernel.phis.items():
                phi_test_norm = np.sqrt(
                    kernel.dot(phi_test[weight], phi_test[weight])
                )
                phis_norms = np.array(
                    [np.sqrt(kernel.dot(phi_tr, phi_tr)) for phi_tr in phis]
                )
                dots = np.array(
                    [kernel.dot(phi_test[weight], phi_tr) for phi_tr in phis]
                )
                Kx += weight * dots / (phis_norms * phi_test_norm)

            pred = (
                np.dot(fitter.alpha * fitter.sv_label, Kx[fitter.sv_indices])
                + fitter.intercept
            )
            predictions.append(np.sign(pred))

if __name__ == "__main__":
    from src.dataset import Dataset
    dataset = Dataset(1)
    kernel_params = {'name': 'mismatch', 'k': 9, 'm': 1}
    fitter_params = {'name': 'svm', 'C': 1, 'tol': 1e-4}
    verbose = True
    kernel, fitter = get_obj(dataset, kernel_params, fitter_params, verbose)

