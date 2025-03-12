import numpy as np
from collections import defaultdict

from src.dataset import Dataset
from src.kernel import WeightedSumKernel
from src.fitter import SVM
from src.crossvalid import CrossValid
from src.utils import write_results, load_experiments
from src.config import Config

class EnsembleSearch:
    def __init__(self, datasets, top_n=10, ensemble_weights=None):
        self.datasets = datasets
        self.top_n = top_n
        self.ensemble_weights = ensemble_weights
        self.best_configs = {}
    
    def find_best_params(self):
        print("Finding best parameter configurations for each dataset...")
        
        experiments = load_experiments(Config.experiments_dir)
        
        dataset_experiments = defaultdict(list)
        for exp in experiments:
            dataset_idx = exp.get("dataset")
            if dataset_idx is not None:
                dataset_experiments[dataset_idx].append(exp)
        
        self.best_configs = {}
        for dataset_idx in self.datasets:
            sorted_experiments = sorted(
                dataset_experiments.get(dataset_idx, []),
                key=lambda x: x.get("results", 0),
                reverse=True
            )
                
            self.best_configs[dataset_idx] = sorted_experiments[:self.top_n]
                    
        return self.best_configs
    
    def _normalize_weights(self, configs):
        print(configs)
        if self.ensemble_weights and len(self.ensemble_weights) == len(configs):
            total_weight = sum(self.ensemble_weights)
            for i, config in enumerate(configs):
                config["kernel"]["weight"] = self.ensemble_weights[i] / total_weight
        else:
            accuracies = np.array([config["results"] for config in configs])
            exp_accs = np.exp(accuracies - np.max(accuracies))
            weights = exp_accs / np.sum(exp_accs)
            for i, config in enumerate(configs):
                config["kernel"]["weight"] = weights[i]
        return configs
    
    def run_search(self, cv_k=5, verbose=True):
        if not self.best_configs:
            self.find_best_params()
        
        results = {}
        
        for dataset_idx in self.datasets:
            if dataset_idx not in self.best_configs or not self.best_configs[dataset_idx]:
                print(f"No configurations found for dataset {dataset_idx}, skipping...")
                continue
                
            print(f"\nRunning ensemble for dataset {dataset_idx}...")
            
            dataset = Dataset(dataset_idx)
            
            configs = self._normalize_weights(self.best_configs[dataset_idx])
            
            kernel_configs = [config["kernel"] for config in configs]
            kernel = WeightedSumKernel(dataset, kernel_configs, verbose=verbose, cache=True)
            
            fitter = SVM(C=1, tol=1e-4)
            
            cross_valid = CrossValid(fitter, dataset, kernel, weight=True, k=cv_k, verbose=verbose)
            fold_results, cv_acc = cross_valid.fit()
            
            results[dataset_idx] = {
                "ensemble_accuracy": cv_acc,
                "fold_results": fold_results,
                "ensemble_config": kernel_configs
            }
            
            best_individual = self.best_configs[dataset_idx][0]["results"]
            improvement = cv_acc - best_individual

            if improvement > -100:
                write_results(dataset, fitter, kernel, cv_acc, weight=True)
            
            print(f"Dataset {dataset_idx} results:")
            print(f"  Best individual model accuracy: {best_individual:.4f}")
            print(f"  Ensemble model accuracy: {cv_acc:.4f}")
            print(f"  Improvement: {improvement:.4f} ({improvement/best_individual*100:.2f}%)")
        
        return results


if __name__ == "__main__":
    datasets = [0, 1, 2]
    ensemble = EnsembleSearch(datasets, top_n=1)
    results = ensemble.run_search(cv_k=5, verbose=True)