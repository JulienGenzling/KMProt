import json
import numpy as np
import pandas as pd
from src.dataset import Dataset
from tqdm import tqdm
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
            fitter, kernel = get_obj(dataset, kernel_params, fitter_params, verbose=True)
        else:
            kernel_configs = config[datafold]["kernel_configs"]
            fitter, kernel = get_obj_ensemble(dataset, kernel_configs, fitter_params, verbose=True)

        K = kernel[:, :]
        K_norm = kernel.normalize(K)
        fitter.fit(K_norm, dataset.labels)

        data = pd.read_csv(f"data/Xte{datafold}.csv")
        data = data.sort_values(by="Id")
        sequences = data["seq"].values

        predictions = []
        for seq in tqdm(sequences, desc=f"Processing fold {datafold}"):
            phi = kernel._get_phi(seq)
            Kx = np.zeros(K.shape[0])
            if not ensemble:
                for j, phi_tr in enumerate(kernel.phis):
                    Kx[j] = kernel.dot(phi, phi_tr) / np.sqrt(
                        kernel.dot(phi_tr, phi_tr) * kernel.dot(phi, phi)
                    )
            else:
                for weight, phis in kernel.phis.items():
                    for j, phi_tr in enumerate(phis):
                        Kx[j] += weight * kernel.dot(phi[weight], phi_tr) / np.sqrt(
                            kernel.dot(phi_tr, phi_tr) * kernel.dot(phi[weight], phi[weight])
                        )
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
    all_predictions_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
