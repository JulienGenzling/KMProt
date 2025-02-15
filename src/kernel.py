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
    def dot(x, y):
        x, y = (x, y) if len(x) < len(y) else (y, x)
        res = 0
        for key, val in x.items():
            res += y.get(key, 0) * val
        return res

    def _compute_phi(self, x):
        phi = {}
        for k in range(self.params["kmin"], self.params["kmax"] + 1):
            for offset in range(len(x) - k + 1):
                xkmer = x[offset : offset + k]
                phi[xkmer] = phi.get(xkmer, 0) + 1
        return phi

    def _compute_phis(self):
        phis = []
        for seq in tqdm(self.dataset.sequences, desc="Computing phis"):
            phi = self._compute_phi(seq)
            phis.append(phi)
        return np.array(phis)

    def compute_gram_matrix(self):
        self.phis = self._compute_phis()

        filename = f"multispectrumkernel_{self.params['kmin']}_{self.params['kmax']}.pkl"

        # Check if the file exists
        if os.path.exists(filename):
            print(f"Loading Gram matrix from file: {filename}")
            with open(filename, 'rb') as file:
                self.K = pickle.load(file)
        else:
            print(f"Computing Gram matrix and saving to file: {filename}")
            K = np.zeros((self.n, self.n))
            for i in tqdm(range(self.n), desc="Gram matrix of spectral"):
                for j in range(i, self.n):
                    K[i, j] = K[j, i] = self.dot(self.phis[i], self.phis[j])

            self.K = K
            # Save the computed Gram matrix to a file
            with open(filename, 'wb') as file:
                pickle.dump(self.K, file)

        return self.K

    @staticmethod
    def normalize(K):
        K_norm = K.copy()
        for i in range(K_norm.shape[0]):
            for j in range(i + 1, K_norm.shape[0]):
                q = np.sqrt(K_norm[i, i] * K_norm[j, j])
                if q > 0:
                    K_norm[i, j] /= q
                    K_norm[j, i] = K_norm[i, j]
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
