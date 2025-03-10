import os
import json
import numpy as np
from src.dataset import KFold
from src.utils import write_results

class CrossValid:
    def __init__(self, fitter, dataset, kernel, weight=False, k=5, verbose=False):
        self.fitter = fitter
        self.dataset = dataset
        self.kernel = kernel
        self.k = k
        self.verbose = verbose
        self.weight = weight

    def fit(self):
        kfold = KFold(self.dataset, self.k)
        results = []
        for fold_idx, fold in enumerate(kfold):
            train_dataset = fold["train"]
            test_dataset = fold["test"]
            idx_tr = train_dataset["indices"]
            idx_ts = test_dataset["indices"]

            K_tr = self.kernel[np.ix_(idx_tr, idx_tr)]
            K_ts = self.kernel[np.ix_(idx_tr, idx_ts)]
            K_tr_norm = self.kernel.normalize(K_tr)
            norms_tr = self.kernel.get_norms(idx_tr, self.kernel.phis)
            norms_ts = self.kernel.get_norms(idx_ts, self.kernel.phis)

            self.fitter.fit(
                K_tr_norm, train_dataset["labels"]
            )  # -> creates alpha and intercept

            predictions = self.fitter.predict(K_ts, norms_tr, norms_ts)
            accuracy = np.mean(predictions == test_dataset["labels"])
            results.append({"fold": fold_idx + 1, "accuracy": accuracy})

            if self.verbose:
                print(f"Fold {fold_idx + 1}/{self.k} - Accuracy: {accuracy:.4f}")

        cv_acc = np.mean([result["accuracy"] for result in results])
        if self.verbose:
            print("Overall accuracy : ", cv_acc)

        if not self.weight:
            write_results(
                self.dataset, self.fitter, self.kernel, cv_acc, weight=self.weight
            )

        return results, cv_acc


if __name__ == "__main__":
    from src.dataset import Dataset
    from src.kernel import MultiSpectrumKernel, MismatchKernel
    from src.fitter import SVM

    dataset = Dataset(k=1)
    # dataset.sequences = np.array(["ATCT", "ATTT", "CGTA", "CTCT", "CTTC"])
    # dataset.labels = np.array([0, 1, 1, 1, 0])
    params = {'name': 'mismatch', 'k': 15, 'm': 3}
    kernel = MismatchKernel(dataset, **params, verbose=True)
    fitter = SVM(C=1, tol=1e-4)
    cross_valid = CrossValid(fitter, dataset, kernel, k=5)
    results = cross_valid.fit()
    print("Cross-validation results:", results)
