import numpy as np
from src.dataset import KFold

class CrossValid:
    def __init__(self, fitter, dataset, kernel, k=5):
        self.fitter = fitter
        self.dataset = dataset
        self.kernel = kernel
        self.k = k

    def fit(self):
        kfold = KFold(self.dataset, self.k)
        results = []

        for fold_idx, fold in enumerate(kfold):
            train_dataset = fold['train']
            test_dataset = fold['test']
            idx_tr = train_dataset['indices']
            idx_ts = test_dataset['indices']

            K_tr = self.kernel[np.ix_(idx_tr, idx_tr)]
            K_ts = self.kernel[np.ix_(idx_tr, idx_ts)]
            K_tr_norm = self.kernel.normalize(K_tr)

            self.fitter.fit(K_tr_norm, train_dataset['labels'])

            predictions = self.fitter.predict(K_ts)
            accuracy = np.mean(predictions == test_dataset['labels'])

            results.append({
                'fold': fold_idx + 1,
                'accuracy': accuracy
            })
            print(f"Fold {fold_idx + 1}/{self.k} - Accuracy: {accuracy:.4f}")

        return results


if __name__ == "__main__":
    from src.dataset import Dataset 
    from src.kernel import MultiSpectrumKernel
    from src.fitter import SVM

    dataset = Dataset(k=0)
    # dataset.sequences = np.array(["ATCT", "ATTT", "CGTA", "CTCT", "CTTC"])
    # dataset.labels = np.array([0, 1, 1, 1, 0])
    kernel = MultiSpectrumKernel(dataset)
    fitter = SVM(C=5)
    cross_valid = CrossValid(fitter, dataset, kernel, k=5)
    results = cross_valid.fit()
    print("Cross-validation results:", results)
