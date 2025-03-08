import os
import numpy as np
from collections import Counter, defaultdict
from itertools import product
from tqdm import tqdm
import pickle
import pandas as pd


class Kernel:
    def __init__(self, verbose=False):
        self.K = None
        self.verbose = verbose

    def compute_gram_matrix(self, verbose=False):
        raise NotImplementedError("This method should be overridden by subclasses")

    def __getitem__(self, index):
        if self.K is None:
            self.compute_gram_matrix()
        i, j = index
        return self.K[i, j]

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


class WeightedSumKernel(Kernel):
    def __init__(self, dataset, kernel_params_list, verbose=False):
        super().__init__(verbose=verbose)
        self.dataset = dataset
        self.n = len(self.dataset)
        self.kernel_params_list = kernel_params_list
        self.phis = {}
        self.compute_gram_matrix()

    def compute_gram_matrix(self):
        self.K = np.zeros((self.n, self.n))
        for kernel_params in self.kernel_params_list:
            if kernel_params["name"] == "spectrum":
                kernel = MultiSpectrumKernel(self.dataset, **kernel_params)
                self.K += (
                    kernel_params["weight"]
                    * kernel.K
                )
                self.phis[kernel_params["weight"]] = kernel.phis
            elif kernel_params["name"] == "mismatch":
                kernel = MismatchKernel(self.dataset, **kernel_params)
                self.K += (
                    kernel_params["weight"]
                    * kernel.K
                )
                self.phis[kernel_params["weight"]] = kernel.phis
        return self.K
    
    @staticmethod
    def dot(phi1, phi2):
        if len(phi1) > len(
            phi2
        ):  # Optimize by using the smaller dictionary for iteration
            phi1, phi2 = phi2, phi1

        dotprod = 0
        for subset, count in phi1.items():
            dotprod += phi2.get(subset, 0) * count
        return dotprod
    
    @classmethod
    def get_norms(cls, idx, phis):
        norms = np.zeros(len(idx))
        for weight, phis_ in phis.items():
            norms += weight * np.array([np.sqrt(cls.dot(phis_[i], phis_[i])) for i in idx])
        return norms

class KmersKernels(Kernel):

    def __init__(self, dataset, verbose=False, **params):
        super().__init__(verbose=verbose)
        self.dataset = dataset
        self.n = len(self.dataset)
        self.params = {}  # Default params to be overridden by subclasses
        self.params.update(params)

    @staticmethod
    def dot(phi1, phi2):
        if len(phi1) > len(
            phi2
        ):  # Optimize by using the smaller dictionary for iteration
            phi1, phi2 = phi2, phi1

        dotprod = 0
        for subset, count in phi1.items():
            dotprod += phi2.get(subset, 0) * count
        return dotprod

    def _get_phi(self, seq):
        raise NotImplementedError("This method should be overridden by subclasses")

    def _get_phis(self):
        phis = []
        for seq in tqdm(self.dataset.sequences, desc="Computing phis...", disable=not self.verbose):
            phi = self._get_phi(seq)
            phis.append(phi)
        return np.array(phis)

    def compute_gram_matrix(self):
        self.phis = self._get_phis()

        filename = self._get_cache_filename()

        # Check if the file exists
        if os.path.exists(filename):
            if self.verbose:
                print(f"Loading Gram matrix from file: {filename}")
            with open(filename, "rb") as file:
                self.K = pickle.load(file)
        else:
            if self.verbose:
                print(f"Computing Gram matrix and saving to file: {filename}")
            K = np.zeros((self.n, self.n))
            for i in tqdm(range(self.n), desc="Gram matrix of spectral", disable=not self.verbose):
                for j in range(i, self.n):
                    K[i, j] = K[j, i] = self.dot(self.phis[i], self.phis[j])

            self.K = K
            # Save the computed Gram matrix to a file
            with open(filename, "wb") as file:
                pickle.dump(self.K, file)

        return self.K

    def _get_cache_filename(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    @classmethod
    def get_norms(cls, idx, phis):
        norms = np.array([np.sqrt(cls.dot(phis[i], phis[i])) for i in idx])
        return norms


class MultiSpectrumKernel(KmersKernels):
    """Kernel that counts exact matches of varying-length k-mers."""

    def __init__(self, dataset, verbose=False, **params):
        """
        Initialize the MultiSpectrum Kernel.

        Parameters:
        dataset: Dataset object containing sequences
        params: Dictionary with kernel parameters:
            - kmin: Minimum length of k-mers (default: 7)
            - kmax: Maximum length of k-mers (default: 20)
        """
        self.params = {"kmin": 7, "kmax": 20}  # Default params
        super().__init__(dataset, verbose=verbose, **params)
        if self.verbose:
            print(f"MultiSpectrumKernel params: {self.params}")

        self.compute_gram_matrix()

    def _get_phi(self, seq):
        phi = {}
        for k in range(self.params["kmin"], self.params["kmax"] + 1):
            for start_index in range(len(seq) - k + 1):
                xkmer = seq[start_index : start_index + k]
                phi[xkmer] = phi.get(xkmer, 0) + 1
        return phi

    def _get_cache_filename(self):
        return f"/Data/kmprotdata/multispectrumkernel_{self.dataset.k}_{self.params['kmin']}_{self.params['kmax']}.pkl"


class MismatchKernel(KmersKernels):
    """Kernel that allows for mismatches when counting k-mers."""

    def __init__(self, dataset, verbose=False, **params):
        """
        Initialize the Mismatch Kernel.

        Parameters:
        dataset: Dataset object containing sequences
        params: Dictionary with kernel parameters:
            - k: Length of k-mers (default: 7)
            - m: Maximum number of mismatches allowed (default: 1)
        """
        self.params = {"k": 7, "m": 1}  # Default params
        super().__init__(dataset, verbose=verbose, **params)
        if self.verbose:
            print(f"MismatchKernel params: {self.params}")

        self._neighborhood_cache = {}
        self.alphabet = ["A", "C", "G", "T"]

        self.compute_gram_matrix()

    def _generate_mismatch_neighborhood(self, kmer, m):
        cache_key = (kmer, m)
        if cache_key in self._neighborhood_cache:
            return self._neighborhood_cache[cache_key]

        if m == 0:
            return {kmer}

        neighborhood = {kmer}

        for i in range(len(kmer)):
            prefix = kmer[:i]
            suffix = kmer[i + 1 :] if i < len(kmer) - 1 else ""
            for base in self.alphabet:
                if base != kmer[i]:
                    variant = prefix + base + suffix
                    neighborhood.add(variant)

        if m > 1:
            extended_neighborhood = set(neighborhood)
            for variant in neighborhood:
                if variant != kmer:
                    extended_neighborhood.update(
                        self._generate_mismatch_neighborhood(variant, m - 1)
                    )
            neighborhood = extended_neighborhood

        self._neighborhood_cache[cache_key] = neighborhood
        return neighborhood

    def _get_phi(self, seq):
        k = self.params["k"]
        m = self.params["m"]

        kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]

        phi = defaultdict(int)

        for kmer in kmers:
            neighborhood = self._generate_mismatch_neighborhood(kmer, m)
            for neighbor in neighborhood:
                phi[neighbor] += 1

        return dict(phi)

    def _get_cache_filename(self):
        return (
            f"/Data/kmprotdata/mismatchkernel_{self.dataset.k}_{self.params['k']}_{self.params['m']}.pkl"
        )


if __name__ == "__main__":
    from src.dataset import Dataset

    # Example usage
    dataset = Dataset(0)

    # Test the spectrum kernel
    spectrum_kernel = MultiSpectrumKernel(dataset, kmin=7, kmax=20)
    print("Spectrum kernel matrix shape:", spectrum_kernel.K.shape)

    # Test the mismatch kernel
    mismatch_kernel = MismatchKernel(dataset, k=5, m=1)
    print("Mismatch kernel matrix shape:", mismatch_kernel.K.shape)
