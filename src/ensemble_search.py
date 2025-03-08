import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from src.dataset import Dataset
from src.kernel import WeightedSumKernel, MultiSpectrumKernel, MismatchKernel
from src.fitter import SVM
from src.crossvalid import CrossValid


class EnsembleSearch:
    def __init__(self, datasets, top_n=10, ensemble_weights=None):
        """
        Initialize the EnsembleSearch class.
        
        Parameters:
        datasets (list): List of dataset indices
        top_n (int): Number of top configurations to use per dataset
        ensemble_weights (list, optional): Weights for each kernel in the ensemble.
                                         If None, equal weights will be used.
        """
        self.datasets = datasets
        self.top_n = top_n
        self.ensemble_weights = ensemble_weights
        self.best_configs = {}
        
    def _load_experiments(self):
        """
        Load all experiment files from the experiments directory.
        
        Returns:
        list: List of experiment data dictionaries
        """
        experiments = []
        experiments_dir = "experiments"
        
        for filename in os.listdir(experiments_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(experiments_dir, filename), "r") as f:
                        experiment = json.load(f)
                        experiments.append(experiment)
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        return experiments
    
    def find_best_params(self):
        """
        Explore the experiment files and find the best configurations for each dataset.
        
        Returns:
        dict: Dictionary mapping dataset indices to lists of top kernel configurations
        """
        print("Finding best parameter configurations for each dataset...")
        
        # Load all experiments
        experiments = self._load_experiments()
        
        # Group experiments by dataset
        dataset_experiments = defaultdict(list)
        for exp in experiments:
            dataset_idx = exp.get("dataset")
            if dataset_idx is not None:
                dataset_experiments[dataset_idx].append(exp)
        
        # Find top configurations for each dataset
        self.best_configs = {}
        for dataset_idx in self.datasets:
            # Sort experiments by accuracy (descending)
            sorted_experiments = sorted(
                dataset_experiments.get(dataset_idx, []),
                key=lambda x: x.get("results", 0),
                reverse=True
            )
            
            # Take top N configurations
            top_configs = []
            for i, exp in enumerate(sorted_experiments[:self.top_n]):
                kernel_info = exp.get("kernel", {})
                kernel_params = kernel_info.get("params", {})
                
                # Convert from experiment format to kernel parameters format
                kernel_config = {
                    "type": kernel_params.get("name", "spectrum"),
                    "weight": 1.0,  # Default weight, will be adjusted later
                }
                
                # Add kernel-specific parameters
                if kernel_config["type"] == "spectrum":
                    kernel_config.update({
                        "kmin": kernel_params.get("kmin", 5),
                        "kmax": kernel_params.get("kmax", 15),
                        "verbose": False
                    })
                elif kernel_config["type"] == "mismatch":
                    kernel_config.update({
                        "k": kernel_params.get("k", 5),
                        "m": kernel_params.get("m", 1),
                        "verbose": False
                    })
                
                top_configs.append({
                    "kernel_config": kernel_config,
                    "accuracy": exp.get("results", 0)
                })
                
            self.best_configs[dataset_idx] = top_configs
            
            # Report top configurations
            print(f"\nDataset {dataset_idx} - Top {len(top_configs)} configurations:")
            for i, config in enumerate(top_configs):
                kernel_type = config["kernel_config"]["type"]
                if kernel_type == "spectrum":
                    kernel_info = f"spectrum(kmin={config['kernel_config']['kmin']}, kmax={config['kernel_config']['kmax']})"
                elif kernel_type == "mismatch":
                    kernel_info = f"mismatch(k={config['kernel_config']['k']}, m={config['kernel_config']['m']})"
                else:
                    kernel_info = kernel_type
                print(f"  {i+1}. {kernel_info}: accuracy = {config['accuracy']:.4f}")
        
        return self.best_configs
    
    def _normalize_weights(self, configs):
        """
        Normalize weights for ensemble kernels.
        
        Parameters:
        configs (list): List of kernel configurations with accuracies
        
        Returns:
        list: List of kernel configurations with normalized weights
        """
        # If ensemble_weights is provided, use those weights
        if self.ensemble_weights and len(self.ensemble_weights) == len(configs):
            total_weight = sum(self.ensemble_weights)
            for i, config in enumerate(configs):
                config["kernel_config"]["weight"] = self.ensemble_weights[i] / total_weight
        else:
            # Use accuracies for weighting (softmax)
            accuracies = np.array([config["accuracy"] for config in configs])
            # Apply softmax to convert accuracies to weights
            exp_accs = np.exp(accuracies - np.max(accuracies))  # Subtract max for numerical stability
            weights = exp_accs / np.sum(exp_accs)
            
            for i, config in enumerate(configs):
                config["kernel_config"]["weight"] = weights[i]
        
        return configs
    
    def run_search(self, cv_k=5, verbose=True):
        """
        Run ensemble search using top configurations for each dataset.
        
        Parameters:
        cv_k (int): Number of folds for cross-validation
        verbose (bool): Whether to show detailed output
        
        Returns:
        dict: Results for each dataset
        """
        if not self.best_configs:
            self.find_best_params()
        
        results = {}
        
        for dataset_idx in self.datasets:
            if dataset_idx not in self.best_configs or not self.best_configs[dataset_idx]:
                print(f"No configurations found for dataset {dataset_idx}, skipping...")
                continue
                
            print(f"\nRunning ensemble for dataset {dataset_idx}...")
            
            # Load dataset
            dataset = Dataset(dataset_idx)
            
            # Normalize weights for ensemble
            configs = self._normalize_weights(self.best_configs[dataset_idx])
            
            # Extract kernel configurations
            kernel_configs = [config["kernel_config"] for config in configs]
            
            # Print ensemble configuration
            if verbose:
                print("Ensemble configuration:")
                for i, config in enumerate(kernel_configs):
                    kernel_type = config["type"]
                    weight = config["weight"]
                    if kernel_type == "spectrum":
                        kernel_info = f"spectrum(kmin={config['kmin']}, kmax={config['kmax']})"
                    elif kernel_type == "mismatch":
                        kernel_info = f"mismatch(k={config['k']}, m={config['m']})"
                    else:
                        kernel_info = kernel_type
                    print(f"  {i+1}. {kernel_info}: weight = {weight:.4f}")
            
            # Create weighted ensemble kernel
            kernel = WeightedSumKernel(dataset, kernel_configs, verbose=verbose)
            
            # Create SVM fitter with default parameters
            fitter = SVM(C=1, tol=1e-4)
            
            # Run cross-validation
            cross_valid = CrossValid(fitter, dataset, kernel, k=cv_k, verbose=verbose)
            fold_results, cv_acc = cross_valid.fit()
            
            results[dataset_idx] = {
                "ensemble_accuracy": cv_acc,
                "fold_results": fold_results,
                "ensemble_config": kernel_configs
            }
            
            # Compare with best individual model
            best_individual = self.best_configs[dataset_idx][0]["accuracy"]
            improvement = cv_acc - best_individual
            
            print(f"Dataset {dataset_idx} results:")
            print(f"  Best individual model accuracy: {best_individual:.4f}")
            print(f"  Ensemble model accuracy: {cv_acc:.4f}")
            print(f"  Improvement: {improvement:.4f} ({improvement/best_individual*100:.2f}%)")
        
        return results


if __name__ == "__main__":
    datasets = [0, 1, 2]
    
    # Create and run ensemble search
    ensemble = EnsembleSearch(datasets, top_n=5)
    results = ensemble.run_search(cv_k=5, verbose=True)
    
    # Save results to file
    with open("ensemble_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nEnsemble search completed. Results saved to ensemble_results.json")