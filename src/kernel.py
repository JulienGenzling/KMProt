import numpy as np
from collections import Counter
from itertools import product
from tqdm import tqdm
import logging
import multiprocessing as mp

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


class SpectrumKernel:
    def __init__(self, dataset, **params):
        self.dataset = dataset
        self.params = {"kmin": 7, "kmax": 20}
        self.params.update(params)

        self.K = None
        self.compute_gram_matrix()

    def get_phi_u(self, seq, k, betas):
        return np.array(
            [seq[i : i + k] == beta for beta in betas for i in range(len(seq) - k + 1)],
            dtype=int,
        )

    def compute_gram_matrix(self):
        k = self.params["kmin"]
        seq = self.dataset.sequences
        n = len(seq)
        K = np.zeros((n, n))
        
        betas = ["".join(c) for c in product("ACGT", repeat=k)]

        phi_u = [
            self.get_phi_u(seq, k, betas)
            for seq in tqdm(seq, desc="Computing feature vectors")
        ]

        # Build the kernel matrix
        for i in tqdm(range(n), desc="Building kernel"):
            for j in range(i, n):
                K[i, j] = np.dot(phi_u[i], phi_u[j])
                K[j, i] = K[i, j]

        self.K = K 
        return self.K

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
    kernel = SpectrumKernel(dataset, kmin=2, kmax=3)

    # Accessing the Gram matrix values
    print(kernel.K)  # Full Gram matrix
