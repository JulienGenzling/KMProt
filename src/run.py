import json
import numpy as np
from src.dataset import Dataset
from src.crossvalid import CrossValid
from src.fitter import SVM, KRR, KLR
from src.kernel import MultiSpectrumKernel, MismatchKernel
from src.utils import get_obj


def load_config(file_path):
    with open(file_path, "r") as file:
        config = json.load(file)
    return config


def main():

    np.random.seed(42)

    config = load_config("src/config.json")
    overall_acc = []

    for datafold in config:
        dataset = Dataset(int(datafold))

        kernel_params = config[datafold]["kernel_params"]
        fitter_params = config[datafold]["fitter_params"]

        kernel, fitter = get_obj(kernel_params, fitter_params)

        cross_valid = CrossValid(fitter, dataset, kernel, k=5)
        results, cv_acc = cross_valid.fit()
        overall_acc.append(cv_acc)

    print("Mean accuracy across the 3 datasets : ", np.mean(overall_acc))


if __name__ == "__main__":
    main()
