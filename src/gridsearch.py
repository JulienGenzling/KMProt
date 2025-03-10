import itertools
import time
import logging
import os
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from functools import partial

from src.dataset import Dataset
from src.crossvalid import CrossValid
from src.utils import get_obj


# Define a helper function for unpacking arguments - defined at module level
def _run_task(args):
    """Helper function to unpack arguments for process_parameter_combination"""
    func, arg_tuple = args
    return func(*arg_tuple)


# Setup the logger for the main process
def setup_logger(log_file=None):
    logger = logging.getLogger("HyperGridSearchLogger")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
        
    # Console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if a log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


# Function to process one parameter combination - defined at module level
def process_parameter_combination(
    dataset_idx, 
    kernel_params_orig, 
    fitter_params_orig, 
    cv_k, 
    experiments_dir="experiments"
):
    # Set up both file and console logging for each process
    process_id = multiprocessing.current_process().name
    logger = logging.getLogger(f"Process-{process_id}")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create a console handler for immediate output
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(processName)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create a file handler for this process to capture detailed logs
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(f"logs/process_{process_id}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Processing dataset {dataset_idx} with kernel {kernel_params_orig} and fitter {fitter_params_orig}")

    kernel_params = kernel_params_orig.copy()
    fitter_params = fitter_params_orig.copy()

    try:
        logger.info(f"Creating dataset {dataset_idx}")
        dataset = Dataset(dataset_idx)
        
        logger.info(f"Creating kernel and fitter objects")
        kernel, fitter = get_obj(dataset, kernel_params, fitter_params, verbose=False)
        
        logger.info(f"Setting up cross-validation with k={cv_k}")
        cross_valid = CrossValid(fitter, dataset, kernel, k=cv_k, verbose=False)

        logger.info(f"Starting cross-validation")
        results, _ = cross_valid.fit()  # This will write results to experiments.json
        
        # Save results manually to ensure they're properly written
        result_dict = {
            "dataset": dataset_idx,
            "kernel": kernel_params,
            "fitter": fitter_params,
            "results": results
        }
        
        # Ensure experiments directory exists
        os.makedirs(experiments_dir, exist_ok=True)
        
        # Create a unique filename
        timestamp = int(time.time())
        filename = f"{experiments_dir}/experiment_{dataset_idx}_{kernel_params['name']}_{fitter_params['name']}_{timestamp}.json"
        
        logger.info(f"Saving results to {filename}")
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2)
            
        logger.info(f"✅ SUCCESSFULLY processed dataset {dataset_idx} with kernel {kernel_params} and fitter {fitter_params}")
        return result_dict
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(
            f"❌ ERROR processing dataset {dataset_idx} with kernel {kernel_params} and fitter {fitter_params}: {str(e)}\n{error_trace}"
        )
        return None


class HyperGridSearch:
    def __init__(
        self,
        datasets,
        kernel_param_grid,
        fitter_param_grid,
        experiments_dir="experiments",
        log_file="hypergrid_search.log"
    ):
        self.datasets = datasets
        self.kernel_param_grid = kernel_param_grid
        self.fitter_param_grid = fitter_param_grid
        self.experiments_dir = experiments_dir
        self.log_file = log_file
        self.logger = setup_logger(log_file)  # Initialize logger with file output
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
        total_combinations = 0
        skipped_combinations = 0
        
        # Corrected logic for checking existing experiments
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
        self.logger.info(f"Skipping {skipped_combinations} already processed combinations")
        self.logger.info(f"Processing {len(all_combinations)} combinations")

        remaining_combinations = len(all_combinations)
        
        if remaining_combinations == 0:
            self.logger.info("No combinations to process. All experiments already exist.")
            return []

        # Create a partial function with the experiments_dir parameter
        process_func = partial(
            process_parameter_combination,
            experiments_dir=self.experiments_dir
        )
        
        # Prepare the arguments for _run_task
        task_args = []
        for dataset_idx, kernel_params, fitter_params in all_combinations:
            arg_tuple = (dataset_idx, kernel_params, fitter_params, cv_k)
            task_args.append((process_func, arg_tuple))

        start_time = time.time()

        # Set the start method for multiprocessing
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # If already set, ignore
            pass
        
        # Process the combinations
        results = []
        with Pool(processes=max_workers) as pool:
            if verbose:
                pbar = tqdm(total=remaining_combinations, desc="Processing combinations")

            # Map to the unpacking function that is defined at module level
            for result in pool.imap_unordered(_run_task, task_args):
                results.append(result)
                if verbose and result is not None:
                    pbar.update(1)

            if verbose:
                pbar.close()
                
            # Ensure all processes terminate properly
            pool.close()
            pool.join()

        end_time = time.time()
        self.logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
        
        # Return non-None results
        return [r for r in results if r is not None]


if __name__ == "__main__":
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("main_process.log"),
            logging.StreamHandler()
        ]
    )
    
    # Clean up any existing log files
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Set number of processes explicitly to avoid exceeding your limit
    max_workers = 10
    print(f"Setting up grid search with {max_workers} worker processes")
    
    datasets = [0, 1, 2]
    kernel_param_grid = {
        "name": ["spectrum", "mismatch"],
        "kmin": range(5, 21),
        "kmax": range(5, 21),
        "k": range(5, 15),
        "m": range(1, 4),
    }
    fitter_param_grid = {
        "name": ["svm"],
        "C": [1],
        "tol": [1e-4],
    }

    search = HyperGridSearch(
        datasets, 
        kernel_param_grid, 
        fitter_param_grid,
        log_file="hypergrid_search.log"  # Add a log file to keep all logs
    )

    # Explicitly set multiprocessing start method before creating the search object
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Method already set
        pass
        
    # Print CPU count to verify
    print(f"System CPU count: {multiprocessing.cpu_count()}")
    print(f"Using workers: {max_workers}")

    results = search.run_search(
        cv_k=5,
        max_workers=max_workers,  # Fixed value from above
        skip_existing=True,  # Skip already processed experiments
    )
    
    print(f"Grid search complete. Processed {len(results)} combinations successfully.")