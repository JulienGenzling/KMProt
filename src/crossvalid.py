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

            self.fitter.fit(
                K_tr, train_dataset["labels"]
            )  # -> creates alpha and intercept

            predictions = self.fitter.predict(K_ts)
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
    from src.kernel import MultiSpectrumKernel, MismatchKernel, WeightedSumKernel
    from src.fitter import SVM

    all_acc = []
    for i in range(10):
        dataset = Dataset(k=0)
        params = {'name': 'spectrum', 'kmin': 5, 'kmax': 9}
        kernel = MultiSpectrumKernel(dataset, **params, verbose=True)
        fitter = SVM(C=1, tol=1e-4)
        cross_valid = CrossValid(fitter, dataset, kernel, k=5)
        results, acc = cross_valid.fit()
        print(acc)
        all_acc.append(acc)
    print(np.mean(all_acc), " +- ", np.std(all_acc))