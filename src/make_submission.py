import json
import numpy as np
import pandas as pd
from src.dataset import Dataset
from src.fitter import SVM
from src.kernel import MultiSpectrumKernel
from tqdm import tqdm


def load_config(file_path):
    with open(file_path, "r") as file:
        config = json.load(file)
    return config


def main():
    config = load_config("src/config.json")
    all_predictions = []

    for datafold in config:
        dataset = Dataset(int(datafold))

        kernel_params = config[datafold]["kernel_params"]
        fitter_params = config[datafold]["fitter_params"]

        if kernel_params["name"] == "spectrum":
            kernel = MultiSpectrumKernel(dataset, **kernel_params)
        else:
            print("Kernel not implemented")
            continue

        if fitter_params["name"] == "svm":
            fitter = SVM(**fitter_params)
        else:
            print("Fitter not implemented")
            continue

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
            for j, phi_tr in enumerate(kernel.phis):
                Kx[j] = kernel.dot(phi, phi_tr) / np.sqrt(
                    kernel.dot(phi_tr, phi_tr) * kernel.dot(phi, phi)
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
    all_predictions_df.to_csv("submission_all.csv", index=False)


if __name__ == "__main__":
    main()
