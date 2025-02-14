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

            self.K = self.normalize(K)

            # Save the computed Gram matrix to a file
            with open(filename, 'wb') as file:
                pickle.dump(self.K, file)

        return self.K

    # def compute_partial_gram_matrix(self, indices, phis):
    #     partial_K = np.zeros((len(indices), len(indices)))
    #     for i, idx_i in enumerate(indices):
    #         for j, idx_j in enumerate(indices):
    #             if idx_i <= idx_j:
    #                 partial_K[i, j] = partial_K[j, i] = self.dot(
    #                     phis[idx_i], phis[idx_j]
    #                 )
    #     return partial_K

    # def compute_gram_matrix(self):
    #     phis = self._compute_phis()
    #     K = np.zeros((self.n, self.n))

    #     num_processes = min(cpu_count(), 8)
    #     chunk_size = self.n // num_processes
    #     chunks = [
    #         list(range(i * chunk_size, (i + 1) * chunk_size))
    #         for i in range(num_processes)
    #     ]
    #     chunks[-1] = list(range(chunks[-1][0], self.n))

    #     with Pool(num_processes) as pool:
    #         results = list(
    #             tqdm(
    #                 pool.starmap(
    #                     self.compute_partial_gram_matrix,
    #                     [(chunk, phis) for chunk in chunks],
    #                 ),
    #                 total=num_processes,
    #                 desc="Computing gram matrix",
    #             )
    #         )

    #     for i, chunk in enumerate(chunks):
    #         for j, idx in enumerate(chunk):
    #             K[idx, chunk] = results[i][j]
    #             K[chunk, idx] = results[i][:, j]

    # self.K = self.normalize(K)
    # return self.K

    @staticmethod
    def normalize(K):
        for i in range(K.shape[0]):
            for j in range(i + 1, K.shape[0]):
                q = np.sqrt(K[i, i] * K[j, j])
                if q > 0:
                    K[i, j] /= q
                    K[j, i] = K[i, j]
        np.fill_diagonal(K, 1.0)
        return K

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
