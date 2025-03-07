import numpy as np
import pandas as pd
import itertools
import os
import json
from datetime import datetime
from tqdm import tqdm
import concurrent.futures
import threading

from src.dataset import Dataset
from src.crossvalid import CrossValid
from src.utils import get_obj


class HyperGridSearch:
    def __init__(self, datasets, kernel_param_grid, fitter_param_grid):
        self.datasets = datasets
        self.kernel_param_grid = kernel_param_grid
        self.fitter_param_grid = fitter_param_grid

    def _generate_param_grid(self, param_grid: dict) -> list:
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    def _process_parameter_combination(self, combo, cv_k):
        dataset_idx, kernel_params_orig, fitter_params_orig = combo

        kernel_params = kernel_params_orig.copy()
        fitter_params = fitter_params_orig.copy()

        dataset = Dataset(dataset_idx)
        kernel, fitter = get_obj(kernel_params, fitter_params)
        cross_valid = CrossValid(fitter, dataset, kernel, k=cv_k)

        _, _ = cross_valid.fit()  # will write results to experiments.json

    def run_search(
        self,
        cv_k: int = 5,
        max_workers: int = 10,
        verbose: bool = True,
    ) -> pd.DataFrame:

        kernel_params_list = self._generate_param_grid(self.kernel_param_grid)
        fitter_params_list = self._generate_param_grid(self.fitter_param_grid)

        all_combinations = []
        for dataset_idx in self.datasets:
            for kernel_params in kernel_params_list:
                for fitter_params in fitter_params_list:
                    all_combinations.append(
                        (dataset_idx, kernel_params.copy(), fitter_params.copy())
                    )

        total_combinations = len(all_combinations)

        if verbose:
            print(
                f"Running parallel grid search with {total_combinations} total combinations"
            )
            print(f"Datasets: {self.datasets}")
            print(f"Kernel parameters: {len(kernel_params_list)} combinations")
            print(f"Fitter parameters: {len(fitter_params_list)} combinations")
            print(f"Using up to {max_workers} threads for parallel processing")

        results_lock = threading.Lock()

        if verbose:
            pbar = tqdm(total=total_combinations, desc="Processing combinations")
            pbar_lock = threading.Lock()

            def update_progress(future):
                with pbar_lock:
                    pbar.update(1)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max_workers, 10)
        ) as executor:
            futures = []
            for combo in all_combinations:
                future = executor.submit(
                    self._process_parameter_combination,
                    combo,
                    cv_k,
                    verbose,
                    results_lock,
                )
                if verbose:
                    future.add_done_callback(update_progress)
                futures.append(future)

            concurrent.futures.wait(futures)

            if verbose:
                pbar.close()

        return pd.DataFrame(self.results)

    @classmethod
    def find_best_params(cls):
        # explores the experiments.json file and gives for each dataset the 10 best configurations
        pass


if __name__ == "__main__":
    datasets = [0, 1, 2]
    kernel_param_grid = {
        "type": ["spectrum", "mismatch"],
        "kmin": range(5, 20),
        "kmax": range(5, 20),
        "k": range(5, 20),
        "m": range(1, 3),
    }
    fitter_param_grid = {
        "type": ["svm"],
        "C": range(1, 5),
        "tol": [1e-4],
    }

    search = HyperGridSearch(datasets, kernel_param_grid, fitter_param_grid)

    results = search.run_search(
        cv_k=5,
        max_workers=10,
    )