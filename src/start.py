import json
import pandas as pd
from src.dataset import Dataset
from src.utils import get_obj, get_obj_ensemble, infer


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
                dataset, kernel_params, fitter_params, verbose=True, cache=False
            )
        else:
            kernel_configs = config[datafold]["kernel_configs"]
            fitter, kernel = get_obj_ensemble(
                dataset, kernel_configs, fitter_params, verbose=True, cache=False
            )

        K = kernel[:, :]
        fitter.fit(K, dataset.labels)

        data = pd.read_csv(f"data/Xte{datafold}.csv")
        data = data.sort_values(by="Id")
        sequences = data["seq"].values

        predictions = infer(kernel, fitter, sequences, ensemble, datafold)
        all_predictions.extend(list(zip(data["Id"], predictions)))

    all_predictions_df = pd.DataFrame(all_predictions, columns=["Id", "Bound"])
    all_predictions_df = all_predictions_df.sort_values(by="Id")
    all_predictions_df["Bound"] = all_predictions_df["Bound"].astype(int)
    all_predictions_df["Bound"] = all_predictions_df["Bound"].map({-1: 0, 1: 1})
    all_predictions_df.to_csv("Yte.csv", index=False)


if __name__ == "__main__":
    main()
