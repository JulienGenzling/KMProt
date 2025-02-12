import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, k):
        if k not in [0, 1, 2]:
            print("Dummy dataset")
        else:
            self.k = k
            self.sequences = self._load_data(f"data/Xtr{k}.csv")
            self.labels = self._load_data(f"data/Ytr{k}.csv", is_label=True)

    def _load_data(self, filename, is_label=False):
        data = pd.read_csv(filename)
        data = data.sort_values(by="Id")

        if is_label:
            labels = data["Bound"].values
            labels_mofidied = np.array([-1 if x==0 else x for x in labels])
            return labels_mofidied
        else:
            return data["seq"].values

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        if index == 0:
            return self.sequences
        elif index == 1:
            return self.labels
        else:
            raise IndexError("Index must be 0 for sequences or 1 for labels")


class KFold:
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k
        self.folds = self._create_folds()

    def _create_folds(self):
        n_samples = len(self.dataset)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        fold_sizes = np.full(self.k, n_samples // self.k, dtype=int)
        fold_sizes[: n_samples % self.k] += 1
        current = 0
        folds = []

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append(indices[start:stop])
            current = stop

        return folds

    def __iter__(self):
        for i in range(self.k):
            test_indices = self.folds[i]
            train_indices = np.concatenate(
                [self.folds[j] for j in range(self.k) if j != i]
            )

            train_dataset = {
                "indices": train_indices,
                "sequences": self.dataset.sequences[train_indices],
                "labels": self.dataset.labels[train_indices],
            }
            test_dataset = {
                "indices": test_indices,
                "sequences": self.dataset.sequences[test_indices],
                "labels": self.dataset.labels[test_indices],
            }

            yield {"train": train_dataset, "test": test_dataset}


# Example usage
if __name__ == "__main__":
    dataset = Dataset(k=0)
    kfold = KFold(dataset, 5)

    for fold in kfold:
        train_dataset = fold["train"]
        print(train_dataset)
        test_dataset = fold["test"]
        print(
            f"Train size: {len(train_dataset['sequences'])}, Test size: {len(test_dataset['sequences'])}"
        )
