import numpy as np
from collections import Counter
from itertools import product
from tqdm import tqdm
import logging
import pickle
from multiprocessing import Pool, cpu_count
import os


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Kernel:
    def __init__(self, dataset):
        self.dataset = dataset
        self.gram_matrix = None

    def compute_gram_matrix(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def __getitem__(self, index):
        if self.gram_matrix is None:
            self.compute_gram_matrix()
        i, j = index
        return self.gram_matrix[i, j]


class MultiSpectrumKernel:
    def __init__(self, dataset, **params):
        self.dataset = dataset
        self.n = len(self.dataset)
        self.params = {"kmin": 7, "kmax": 20}
        self.params.update(params)
        print(self.params)

        self.K = None
        self.compute_gram_matrix()

    @staticmethod
    def dot(phi1, phi2):
        if len(phi1) > len(phi2):
            phi1, phi2 = phi2, phi1
        dotprod = 0
        for subset, count in phi1.items():
            dotprod += phi2.get(subset, 0) * count
        return dotprod

    def _get_phi(self, seq):
        phi = {}
        for k in range(self.params["kmin"], self.params["kmax"] + 1):
            for start_index in range(len(seq) - k + 1):
                xkmer = seq[start_index : start_index + k]
                phi[xkmer] = phi.get(xkmer, 0) + 1
        return phi

    def _get_phis(self):
        phis = []
        for seq in tqdm(self.dataset.sequences, desc="Computing phis..."):
            phi = self._get_phi(seq)
            phis.append(phi)
        return np.array(phis)

    def compute_gram_matrix(self):
        self.phis = self._get_phis()

        filename = (
            f"multispectrumkernel_{self.dataset.k}_{self.params['kmin']}_{self.params['kmax']}.pkl"
        )

        # Check if the file exists
        if os.path.exists(filename):
            print(f"Loading Gram matrix from file: {filename}")
            with open(filename, "rb") as file:
                self.K = pickle.load(file)
        else:
            print(f"Computing Gram matrix and saving to file: {filename}")
            K = np.zeros((self.n, self.n))
            for i in tqdm(range(self.n), desc="Gram matrix of spectral"):
                for j in range(i, self.n):
                    K[i, j] = K[j, i] = self.dot(self.phis[i], self.phis[j])

            self.K = K
            # Save the computed Gram matrix to a file
            with open(filename, "wb") as file:
                pickle.dump(self.K, file)

        return self.K

    @staticmethod
    def normalize(K):
        K_norm = K.copy()
        diag = np.sqrt(np.diag(K_norm))

        with np.errstate(divide="ignore", invalid="ignore"):
            outer_diag = np.outer(diag, diag)
            K_norm /= outer_diag
            K_norm[outer_diag == 0] = 0

        np.fill_diagonal(K_norm, 1.0)
        return K_norm

    @classmethod
    def get_norms(cls, idx, phis):
        norms = np.array([np.sqrt(cls.dot(phis[i], phis[i])) for i in idx])
        return norms

    def __getitem__(self, index):
        if self.K is None:
            self.compute_gram_matrix()
        i, j = index
        return self.K[i, j]


if __name__ == "__main__":
    from src.dataset import Dataset

    # Example usage
    dataset = Dataset(0)
    # dataset = [["ATCT", "ATTT", "CGTA", "CTCT"], [0, 1, 1, 1]]

    kernel = MultiSpectrumKernel(dataset, kmin=7, kmax=20)

    # Accessing the Gram matrix values
    print(kernel.K.shape)
    print(kernel.K)  # Full Gram matrix
