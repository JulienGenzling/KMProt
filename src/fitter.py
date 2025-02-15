import cvxopt
import numpy as np


class Fitter:
    def __init__(self):
        self.intercept = None
        self.alpha = None

    def fit(self, K, y):
        raise NotImplementedError

    def predict(self, K):
        raise NotImplementedError


class SVM(Fitter):
    def __init__(self, **params):
        super().__init__()
        self.params = {"C": 1.0, "tol": 1e-4}
        self.params.update(params)

    def fit(self, K, y):
        """
        Fit the SVM model according to the given kernel matrix K and labels y.
        We solve the SVM dual problem :
        max sum alpha_i - 1/2 sum_{j,k} alpha_j alpha_k y_j y_k K(x_j, x_k)
            st. 0 <= alpha_i <= C and sum alpha_i * y_i = 0

        Parameters:
        K (np.ndarray): Kernel matrix of shape NxN (training samples only)
        y (np.ndarray): Labels.

        Returns:
        None
        """
        n = K.shape[0]
        C = self.params["C"]
        tol = self.params["tol"]

        # Convert inputs to cvxopt matrices
        G = cvxopt.matrix(np.vstack((-np.eye(n), np.eye(n))), tc="d")
        h = cvxopt.matrix(np.hstack((np.zeros(n), C * np.ones(n))), tc="d")
        P = cvxopt.matrix(np.diag(y) @ K @ np.diag(y), tc="d")
        q = cvxopt.matrix(-np.ones(n), tc="d")
        A = cvxopt.matrix(y, (1, n), tc="d")
        b = cvxopt.matrix(0.0, tc="d")

        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        alpha = np.array(sol["x"]).reshape(-1)
        sv_indices = np.where(np.abs(alpha) > tol)[0]
        self.alpha = alpha[sv_indices]
        self.sv_label = y[sv_indices]

        intercept = np.mean(
            [
                y[sv] - np.sum(self.sv_label * self.alpha * K[sv, sv_indices])
                for sv in sv_indices
            ]
        )

        self.intercept = intercept
        self.sv_indices = sv_indices

    def predict(self, K, norms_tr, norms_ts):
        """
        K has shape (N, n) where N are the train samples
        (among which we find the support vectors) and n
        are the test samples.
        """
        print(np.sum(K[:,10]))
        K_norm = K / np.outer(norms_tr, norms_ts)
        y_pred = (
            np.dot(K_norm[self.sv_indices, :].T, self.alpha * self.sv_label)
            + self.intercept
        )
        return np.sign(y_pred)
