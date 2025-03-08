import itertools
import time
import logging
import os
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Manager

from src.dataset import Dataset
from src.crossvalid import CrossValid
from src.utils import get_obj


# Setup the logger
def setup_logger():
    logger = logging.getLogger("HyperGridSearchLogger")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


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
        self.logger = setup_logger()  # Initialize logger
        self.experiments_dir = experiments_dir
        self.existing_experiments = self._load_existing_experiments()

    def _load_existing_experiments(self):
        self.logger.info(
            f"Loading existing experiments from {self.experiments_dir} folder"
        )

        existing_experiments = []
        if os.path.exists(self.experiments_dir):
            for filename in os.listdir(self.experiments_dir):
                if filename.endswith(".json"):
                    try:
                        with open(
                            os.path.join(self.experiments_dir, filename), "r"
                        ) as f:
                            exp = json.load(f)
                        existing_experiments.append(exp)
                    except Exception as e:
                        self.logger.warning(
                            f"Error loading experiment {filename}: {str(e)}"
                        )

        self.logger.info(f"Loaded {len(existing_experiments)} existing experiments")
        return existing_experiments

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

    @staticmethod
    def _process_parameter_combination(args):
        (
            dataset_idx,
            kernel_params_orig,
            fitter_params_orig,
            cv_k,
            logger,
            should_skip,
        ) = args

        if should_skip:
            logger.info(
                f"Skipping already processed dataset {dataset_idx} with kernel {kernel_params_orig} and fitter {fitter_params_orig}"
            )
            return

        kernel_params = kernel_params_orig.copy()
        fitter_params = fitter_params_orig.copy()

        dataset = Dataset(dataset_idx)
        kernel, fitter = get_obj(dataset, kernel_params, fitter_params, verbose=False)
        cross_valid = CrossValid(fitter, dataset, kernel, k=cv_k, verbose=False)

        try:
            _, _ = cross_valid.fit()  # This will write results to experiments.json
            logger.info(
                f"Processed dataset {dataset_idx} with kernel {kernel_params} and fitter {fitter_params}"
            )
        except Exception as e:
            logger.error(
                f"Error processing dataset {dataset_idx} with kernel {kernel_params} and fitter {fitter_params}: {e}"
            )

        return

    def run_search(
        self,
        cv_k: int = 5,
        max_workers: int = None,
        verbose: bool = True,
        skip_existing: bool = True,
    ):
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        else:
            max_workers = min(max_workers, multiprocessing.cpu_count())

        self.logger.info(f"Using {max_workers} processes")

        kernel_params_list = self._generate_kernel_combinations()
        fitter_params_list = self._generate_fitter_combinations()

        self.logger.info(f"Generated {len(kernel_params_list)} kernel configurations")
        self.logger.info(f"Generated {len(fitter_params_list)} fitter configurations")

        all_combinations = []
        skipped_combinations = 0
        for dataset_idx in self.datasets:
            for kernel_params in kernel_params_list:
                for fitter_params in fitter_params_list:
                    should_skip = False
                    if skip_existing and self._experiment_exists(
                        dataset_idx, kernel_params, fitter_params
                    ):
                        should_skip = True
                        skipped_combinations += 1

                    all_combinations.append(
                        (
                            dataset_idx,
                            kernel_params.copy(),
                            fitter_params.copy(),
                            should_skip,
                        )
                    )

        self.logger.info(
            f"Skipping {skipped_combinations} already processed combinations"
        )

        task_args = []
        for dataset_idx, kernel_params, fitter_params, should_skip in all_combinations:
            task_args.append(
                (
                    dataset_idx,
                    kernel_params,
                    fitter_params,
                    cv_k,
                    self.logger,
                    should_skip,
                )
            )

        total_combinations = len(task_args)
        remaining_combinations = total_combinations - skipped_combinations
        self.logger.info(
            f"Total combinations: {total_combinations}, remaining to process: {remaining_combinations}"
        )

        start_time = time.time()

        # Create a pool with start method 'spawn' to avoid issues with pickle
        multiprocessing.set_start_method("spawn", force=True)

        with Pool(processes=max_workers) as pool:
            if verbose:
                pbar = tqdm(
                    total=remaining_combinations, desc="Processing combinations"
                )

            # Use imap_unordered for parallel processing
            for _ in pool.imap_unordered(
                self._process_parameter_combination, task_args
            ):
                if verbose and not _:  # Only update for non-skipped combinations
                    pbar.update(1)

            if verbose:
                pbar.close()

        end_time = time.time()
        self.logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    datasets = [0, 1, 2]
    kernel_param_grid = {
        "name": ["spectrum", "mismatch"],
        "kmin": range(5, 21),
        "kmax": range(5, 21),
        "k": range(5, 21),
        "m": range(1, 4),
    }
    fitter_param_grid = {
        "name": ["svm"],
        "C": [1],
        "tol": [1e-4],
    }

    search = HyperGridSearch(datasets, kernel_param_grid, fitter_param_grid)

    search.run_search(
        cv_k=5,
        max_workers=10,
        skip_existing=True,  # Skip already processed experiments
    )
