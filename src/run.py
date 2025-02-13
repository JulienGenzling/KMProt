import json
import numpy as np
from src.dataset import Dataset
from src.crossvalid import CrossValid
from src.fitter import SVM
from src.kernel import SpectrumKernel


def load_config(file_path):
    with open(file_path, "r") as file:
        config = json.load(file)
    return config


def main():
    config = load_config("src/config.json")
    for datafold in config:
        dataset = Dataset(int(datafold))

        kernel_params = config[datafold]["kernel_params"]
        fitter_params = config[datafold]["fitter_params"]

        if kernel_params["name"] == "spectrum":
            kernel = SpectrumKernel(
                dataset, **kernel_params
            )
        else:
            print("Kernel not implemented")
        if fitter_params["name"] == "svm":
            fitter = SVM(**fitter_params)
        else:
            print("Fitter not implemented")

        cross_valid = CrossValid(fitter, dataset, kernel, k=5)
        results = cross_valid.fit()

        print(f"Results for datafold {datafold}: {results}")


if __name__ == "__main__":
    main()
