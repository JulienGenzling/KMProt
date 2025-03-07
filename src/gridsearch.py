import itertools
import time
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Manager

from src.dataset import Dataset
from src.crossvalid import CrossValid
from src.utils import get_obj


class HyperGridSearch:
    def __init__(self, datasets, kernel_param_grid, fitter_param_grid):
        self.datasets = datasets
        self.kernel_param_grid = kernel_param_grid
        self.fitter_param_grid = fitter_param_grid

    def _generate_kernel_combinations(self):
        all_kernel_combinations = []
        
        if "spectrum" in self.kernel_param_grid["name"]:
            kmin_values = self.kernel_param_grid["kmin"]
            kmax_values = self.kernel_param_grid["kmax"]
            
            for kmin in kmin_values:
                for kmax in [k for k in kmax_values if k >= kmin]:
                    all_kernel_combinations.append({
                        "name": "spectrum",
                        "kmin": kmin,
                        "kmax": kmax
                    })
        
        if "mismatch" in self.kernel_param_grid["name"]:
            k_values = self.kernel_param_grid["k"]
            m_values = self.kernel_param_grid["m"]
            
            for k in k_values:
                for m in [m_val for m_val in m_values if m_val <= k]:
                    all_kernel_combinations.append({
                        "name": "mismatch",
                        "k": k,
                        "m": m
                    })
        
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
        dataset_idx, kernel_params_orig, fitter_params_orig, cv_k = args

        kernel_params = kernel_params_orig.copy()
        fitter_params = fitter_params_orig.copy()

        dataset = Dataset(dataset_idx)
        kernel, fitter = get_obj(dataset, kernel_params, fitter_params, verbose=False)
        cross_valid = CrossValid(fitter, dataset, kernel, k=cv_k, verbose=False)

        _, _ = cross_valid.fit()  # will write results to experiments.json

    def run_search(
        self,
        cv_k: int = 5,
        max_workers: int = None,
        verbose: bool = True,
    ):

        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        else:
            max_workers = min(max_workers, multiprocessing.cpu_count())
            
        print(f"Using {max_workers} processes")
        
        kernel_params_list = self._generate_kernel_combinations()
        fitter_params_list = self._generate_fitter_combinations()
        
        print(f"Generated {len(kernel_params_list)} kernel configurations")
        print(f"Generated {len(fitter_params_list)} fitter configurations")

        all_combinations = []
        for dataset_idx in self.datasets:
            for kernel_params in kernel_params_list:
                for fitter_params in fitter_params_list:
                    all_combinations.append(
                        (dataset_idx, kernel_params.copy(), fitter_params.copy())
                    )
        
        task_args = []
        for dataset_idx, kernel_params, fitter_params in all_combinations:
            task_args.append((dataset_idx, kernel_params, fitter_params, cv_k))
        
        total_combinations = len(task_args)
        print(f"Total combinations to evaluate: {total_combinations}")
        
        start_time = time.time()
        
        multiprocessing.set_start_method('spawn', force=True)
        
        with Pool(processes=max_workers) as pool:
            if verbose:
                pbar = tqdm(total=total_combinations, desc="Processing combinations")
            
            for _ in pool.imap_unordered(
                self._process_parameter_combination, 
                task_args
            ):
                if verbose:
                    pbar.update(1)
            
            if verbose:
                pbar.close()
        
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

    @classmethod
    def find_best_params(cls):
        # explores the experiments.json file and gives for each dataset the 10 best configurations
        pass


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
    )