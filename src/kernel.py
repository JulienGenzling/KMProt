import os
import numpy as np
from itertools import product
from tqdm import tqdm
import pickle
from scipy import sparse

from src.config import Config


class Kernel:
    def __init__(self, verbose=False):
        self.K = None
        self.verbose = verbose

    def compute_gram_matrix(self, cache):
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
    def __init__(self, dataset, kernel_params_list, verbose=False, cache=True):
        super().__init__(verbose=verbose)
        self.verbose = verbose
        self.dataset = dataset
        self.n = len(self.dataset)
        self.kernel_params_list = kernel_params_list
        self.phis = {}
        self.kernels = []
        self.cache = cache
        self.compute_gram_matrix(cache)

    def compute_gram_matrix(self, cache):
        self.K = np.zeros((self.n, self.n))
        for kernel_params in self.kernel_params_list:
            if kernel_params["name"] == "spectrum":
                kernel = MultiSpectrumKernel(self.dataset, **kernel_params, verbose=self.verbose, cache=cache)
                self.K += (
                    kernel_params["weight"]
                    * kernel.K
                )
                self.kernels.append(kernel)
                self.phis[kernel_params["weight"]] = kernel.phis
            elif kernel_params["name"] == "mismatch":
                kernel = MismatchKernel(self.dataset, **kernel_params, verbose=self.verbose, cache=cache)
                self.K += (
                    kernel_params["weight"]
                    * kernel.K
                )
                self.kernels.append(kernel)
                self.phis[kernel_params["weight"]] = kernel.phis
        return self.K

    def _get_phi(self, seq):
        phi = {}
        for i, kernel_params in enumerate(self.kernel_params_list):
            if kernel_params["name"] == "spectrum":
                phi[kernel_params["weight"]] = self.kernels[i]._get_phi(seq)
            elif kernel_params["name"] == "mismatch":
                phi[kernel_params["weight"]] = self.kernels[i]._get_phi(seq)
        return phi
        
    
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
        self.params = {}
        self.params.update(params)

    @staticmethod
    def dot(phi1, phi2):
        if len(phi1) > len(
            phi2
        ):
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

    def compute_gram_matrix(self, cache):
        self.phis = self._get_phis()

        if cache:
            filename = self._get_cache_filename()
            if os.path.exists(filename):
                if self.verbose:
                    print(f"Loading Gram matrix from file: {filename}")
                with open(filename, "rb") as file:
                    self.K = pickle.load(file)
                    self.phis = self._get_phis()
                return self.K

        if self.verbose:
            print(f"Computing Gram matrix...")
        K = np.zeros((self.n, self.n))
        for i in tqdm(range(self.n), desc="Gram matrix of spectral", disable=not self.verbose):
            for j in range(i, self.n):
                K[i, j] = K[j, i] = self.dot(self.phis[i], self.phis[j])

        self.K = self.normalize(K)
        # Save the computed Gram matrix to a file
        if cache:
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
    def __init__(self, dataset, verbose=False, cache=True, **params):
        self.params = {"kmin": 7, "kmax": 20} 
        self.cache=cache
        super().__init__(dataset, verbose=verbose, **params)
        if self.verbose:
            print(f"MultiSpectrumKernel params: {self.params}")

        self.compute_gram_matrix(self.cache)

    def _get_phi(self, seq):
        phi = {}
        for k in range(self.params["kmin"], self.params["kmax"] + 1):
            for start_index in range(len(seq) - k + 1):
                xkmer = seq[start_index : start_index + k]
                phi[xkmer] = phi.get(xkmer, 0) + 1
        return phi

    def _get_cache_filename(self):
        os.makedirs(Config.kernel_dir, exist_ok=True)
        return f"{Config.kernel_dir}/multispectrumkernel_{self.dataset.k}_{self.params['kmin']}_{self.params['kmax']}.pkl"


class MismatchKernel(KmersKernels):
    def __init__(self, dataset, verbose=False, cache=True, **params):
        self.params = {"k": 7, "m": 1} 
        self.cache = cache
        super().__init__(dataset, verbose=verbose, **params)
        if self.verbose:
            print(f"MismatchKernel params: {self.params}")
        
        self.alphabet = ["A", "C", "G", "T"]
        
        self.kmer_set, self.neighbours = self._precompute_kmer_neighbors()
        
        self.compute_gram_matrix(self.cache)
    
    def _precompute_kmer_neighbors(self):
        all_kmers = set()
        k = self.params["k"]
        
        for seq in self.dataset.sequences:
            for i in range(len(seq) - k + 1):
                all_kmers.add(seq[i:i+k])
        
        kmer_set = {kmer: idx for idx, kmer in enumerate(sorted(all_kmers))}
        
        neighbours = {}
        for kmer in tqdm(all_kmers, desc="Computing neighborhoods", disable=not self.verbose):
            neighbours[kmer] = self._generate_mismatch_neighborhood(kmer, self.params["m"])
        
        if self.verbose:
            print(f"Generated {len(kmer_set)} unique kmers and their neighborhoods")
            
        return kmer_set, neighbours
    
    def _generate_mismatch_neighborhood(self, kmer, m):
        if m == 0:
            return {kmer}
        neighborhood = set()
        positions = list(range(len(kmer)))
        for r in range(1, m + 1):
            for pos_combo in product(positions, repeat=r):
                if len(set(pos_combo)) < len(pos_combo): # otherwise it would be like r=r-1
                    continue
                for base_combo in product(self.alphabet, repeat=r):
                    kmer_list = list(kmer)
                    valid_mismatch = False
                    for i, pos in enumerate(pos_combo):
                        if kmer_list[pos] != base_combo[i]: # we found a new neighbor
                            kmer_list[pos] = base_combo[i]
                            valid_mismatch = True
                    if valid_mismatch:
                        neighborhood.add(''.join(kmer_list))
        neighborhood.add(kmer)
        return neighborhood
    
    def neighbour_phi(self, seq):
        kmer_seq = [seq[j:j + self.params["k"]] for j in range(len(seq) - self.params["k"] + 1)]
        seq_emb = {}
        for kmer in kmer_seq:
            if kmer not in self.neighbours:
                continue
            neigh_kmer = self.neighbours[kmer]
            for neigh in neigh_kmer:
                if neigh not in self.kmer_set:
                    continue
                idx_neigh = self.kmer_set[neigh]
                if idx_neigh in seq_emb:
                    seq_emb[idx_neigh] += 1.
                else:
                    seq_emb[idx_neigh] = 1.
        return seq_emb
    
    def _get_phi(self, seq):
        return self.neighbour_phi(seq)
    
    def compute_gram_matrix(self, cache):
        if cache:
            filename = self._get_cache_filename()
            
            if os.path.exists(filename):
                if self.verbose:
                    print(f"Loading Gram matrix from file: {filename}")
                with open(filename, "rb") as file:
                    self.K = pickle.load(file)
                    self.phis = self._get_phis()
                return self.K
            
        self.phis = self._get_phis()
        
        data, row, col = [], [], []
        for i, phi in enumerate(self.phis):
            data.extend(list(phi.values()))
            row.extend(list(phi.keys()))
            col.extend([i] * len(phi))
        
        X_sm = sparse.coo_matrix((data, (row, col)))
        self.K = (X_sm.T @ X_sm).toarray()
        self.K = self.normalize(self.K)
        
        if cache:
            with open(filename, "wb") as file:
                pickle.dump(self.K, file)
        return self.K

    def _get_cache_filename(self):
        os.makedirs(Config.kernel_dir, exist_ok=True)
        return (
            f"{Config.kernel_dir}/mismatchkernel_{self.dataset.k}_{self.params['k']}_{self.params['m']}.pkl"
        )


if __name__ == "__main__":
    from src.dataset import Dataset

    # Example usage
    dataset = Dataset(0)

    # Test the spectrum kernel
    spectrum_kernel = MultiSpectrumKernel(dataset, kmin=1, kmax=2, verbose=True)
    print("Spectrum kernel matrix shape:", spectrum_kernel.K.shape)

    # Test the mismatch kernel
    mismatch_kernel = MismatchKernel(dataset, k=5, m=1, verbose=True)
    print("Mismatch kernel matrix shape:", mismatch_kernel.K.shape)

    # Test the weighted sum kernel
    wsum_kernel = WeightedSumKernel(
        dataset, [{"name": "spectrum", "kmin": 1, "kmax": 2, "weight": 0.5}, {"name": "mismatch", "k": 5, "m": 1, "weight": 0.5}],
        verbose=True
    )
    print("Weighted sum kernel matrix shape:", mismatch_kernel.K.shape)
