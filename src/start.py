import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.dataset import Dataset
from src.utils import get_obj, get_obj_ensemble


def load_config(file_path):
    with open(file_path, "r") as file:
        config = json.load(file)
    return config


def main():
    config = load_config("src/config.json")
    all_predictions = []
    datasets = ["0", "1", "2"]

    for datafold in datasets:
        dataset = Dataset(int(datafold))

        fitter_params = config[datafold]["fitter_params"]
        ensemble = not config["ensemble"] == "False"

        if not ensemble:
            kernel_params = config[datafold]["kernel_params"]
            fitter, kernel = get_obj(
                dataset, kernel_params, fitter_params, verbose=True
            )
        else:
            kernel_configs = config[datafold]["kernel_configs"]
            fitter, kernel = get_obj_ensemble(
                dataset, kernel_configs, fitter_params, verbose=True
            )

        K = kernel[:, :]
        fitter.fit(K, dataset.labels)

        data = pd.read_csv(f"data/Xte{datafold}.csv")
        data = data.sort_values(by="Id")
        sequences = data["seq"].values

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

        all_predictions.extend(list(zip(data["Id"], predictions)))

    all_predictions_df = pd.DataFrame(all_predictions, columns=["Id", "Bound"])
    all_predictions_df = all_predictions_df.sort_values(by="Id")
    all_predictions_df["Bound"] = all_predictions_df["Bound"].astype(int)
    all_predictions_df["Bound"] = all_predictions_df["Bound"].map({-1: 0, 1: 1})
    all_predictions_df.to_csv("Yte.csv", index=False)


if __name__ == "__main__":
    main()
