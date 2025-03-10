import os
import json
import itertools
import time
import logging
import multiprocessing
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from src.dataset import Dataset
from src.crossvalid import CrossValid
from src.utils import get_obj, load_experiments


def _run_task(args):
    func, arg_tuple = args
    return func(*arg_tuple)


def setup_logger(logger):
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def process_parameter_combination(
    dataset_idx, kernel_params_orig, fitter_params_orig, cv_k
):
    process_id = multiprocessing.current_process().name
    logger = logging.getLogger(f"Process-{process_id}")
    logger = setup_logger(logger)

    kernel_params, fitter_params = kernel_params_orig.copy(), fitter_params_orig.copy()

    try:
        dataset = Dataset(dataset_idx)
        kernel, fitter = get_obj(dataset, kernel_params, fitter_params, verbose=False)
        cross_valid = CrossValid(fitter, dataset, kernel, k=cv_k, verbose=False)
        results, _ = cross_valid.fit()
        logger.info(
            f"✅ SUCCESSFULLY processed dataset {dataset_idx} with kernel {kernel_params} and fitter {fitter_params}"
        )
        return {
            "dataset": dataset_idx,
            "kernel": kernel_params,
            "fitter": fitter_params,
            "results": results,
        }
    except Exception as e:
        import traceback

        logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        return None


class HyperGridSearch:
    def __init__(
        self,
        datasets,
        kernel_param_grid,
        fitter_param_grid,
        experiments_dir="experiments",
    ):
        self.datasets = datasets
        self.kernel_param_grid = kernel_param_grid
        self.fitter_param_grid = fitter_param_grid
        self.experiments_dir = experiments_dir
        logger = logging.getLogger("HyperGridSearchLogger")
        self.logger = setup_logger(logger)
        self.existing_experiments = load_experiments(self.experiments_dir)
        logger.info(f"Loaded {len(self.existing_experiments)} experiments")

    def _experiment_exists(self, dataset_idx, kernel_params, fitter_params):
        mock_experiment = {
            "dataset": dataset_idx,
            "fitter": fitter_params,
            "kernel": kernel_params,
        }

        for exp in self.existing_experiments:
            if all(
                exp.get(key) == mock_experiment.get(key)
                for key in ["dataset", "fitter", "kernel"]
            ):
                return True

        return False

    def _generate_kernel_combinations(self):
        all_kernel_combinations = []

        if "spectrum" in self.kernel_param_grid["name"]:
            kmin_values = self.kernel_param_grid["kmin"]
            kmax_values = self.kernel_param_grid["kmax"]

            for kmin in kmin_values:
                for kmax in [k for k in kmax_values if k >= kmin]:
                    all_kernel_combinations.append(
                        {"name": "spectrum", "kmin": kmin, "kmax": kmax}
                    )

        if "mismatch" in self.kernel_param_grid["name"]:
            k_values = self.kernel_param_grid["k"]
            m_values = self.kernel_param_grid["m"]

            for k in k_values:
                for m in [m_val for m_val in m_values if m_val <= k]:
                    all_kernel_combinations.append({"name": "mismatch", "k": k, "m": m})

        return all_kernel_combinations

    def _generate_fitter_combinations(self):
        all_fitter_combinations = []

        for fitter_name in self.fitter_param_grid["name"]:
            fitter_params = {"name": fitter_name}

            param_keys = [key for key in self.fitter_param_grid.keys() if key != "name"]

            param_values = [self.fitter_param_grid[key] for key in param_keys]
            for combo in itertools.product(*param_values):
                combo_dict = dict(zip(param_keys, combo))
                all_fitter_combinations.append({**fitter_params, **combo_dict})

        return all_fitter_combinations

    def run_search(
        self,
        cv_k: int = 5,
        num_workers: int = None,
        verbose: bool = True,
        skip_existing: bool = True,
    ):

        kernel_params_list = self._generate_kernel_combinations()
        fitter_params_list = self._generate_fitter_combinations()

        self.logger.info(f"Generated {len(kernel_params_list)} kernel configurations")
        self.logger.info(f"Generated {len(fitter_params_list)} fitter configurations")

        all_combinations = []
        total_combinations = 0
        skipped_combinations = 0

        for dataset_idx in self.datasets:
            for kernel_params in kernel_params_list:
                for fitter_params in fitter_params_list:
                    total_combinations += 1
                    should_skip = skip_existing and self._experiment_exists(
                        dataset_idx, kernel_params, fitter_params
                    )

                    if should_skip:
                        skipped_combinations += 1
                    else:
                        all_combinations.append(
                            (
                                dataset_idx,
                                kernel_params.copy(),
                                fitter_params.copy(),
                            )
                        )

        self.logger.info(f"Total combinations: {total_combinations}")
        self.logger.info(
            f"Skipping {skipped_combinations} already processed combinations"
        )
        self.logger.info(f"Processing {len(all_combinations)} combinations")

        remaining_combinations = len(all_combinations)

        if remaining_combinations == 0:
            self.logger.info(
                "No combinations to process. All experiments already exist."
            )
            return []

        process_func = partial(
            process_parameter_combination
        )

        task_args = []
        for dataset_idx, kernel_params, fitter_params in all_combinations:
            arg_tuple = (dataset_idx, kernel_params, fitter_params, cv_k)
            task_args.append((process_func, arg_tuple))

        start_time = time.time()

        results = []
        with Pool(processes=num_workers) as pool:
            if verbose:
                pbar = tqdm(
                    total=remaining_combinations, desc="Processing combinations"
                )

            for result in pool.imap_unordered(_run_task, task_args):
                results.append(result)
                if verbose and result is not None:
                    pbar.update(1)

            if verbose:
                pbar.close()

            pool.close()
            pool.join()

        end_time = time.time()
        self.logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

        return [r for r in results if r is not None]


if __name__ == "__main__":

    max_workers = min(10, multiprocessing.cpu_count())
    print(f"Setting up grid search with {max_workers} worker processes")

    datasets = [0, 1, 2]

    # Don't use high values for (k,m) if mismatch because will exceed RAM quickly
    kernel_param_grid = {
        "name": ["spectrum", "mismatch"],
        "kmin": range(5, 21),
        "kmax": range(5, 21),
        "k": range(5, 15),
        "m": range(1, 3),
    }
    fitter_param_grid = {
        "name": ["svm"],
        "C": [2, 3, 4, 5],
        "tol": [1e-4],
    }

    search = HyperGridSearch(
        datasets,
        kernel_param_grid,
        fitter_param_grid,
    )

    results = search.run_search(
        cv_k=5,
        num_workers=max_workers,
    )

    print(f"Grid search complete. Processed {len(results)} combinations successfully.")
