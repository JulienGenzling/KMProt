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
            self.sv_label
            - self.sv_label * np.dot(K[sv_indices][:, sv_indices], self.alpha)
        )

        self.intercept = intercept
        self.sv_indices = sv_indices

        return
    # def fit(self, K, y):
    #     print("3OHIOHEIDUDBEIDUBIZEUBCUEIBCIEZUCBIEUZBCIUEZBCUI")
    #     print(K)
    #     n = K.shape[0]
    #     C = self.params["C"]
    #     tol = self.params["tol"]

    #     G_top = np.diag(np.ones(n) * (-1))
    #     h_left = np.zeros(n)
    #     G_bot = np.eye(n)
    #     h_right = np.ones(n) * C
    #     G = cvxopt.matrix(np.vstack([G_top, G_bot]), (2 * n, n), 'd')
    #     h = cvxopt.matrix(np.hstack([h_left, h_right]), (2 * n, 1), 'd')
    #     P = cvxopt.matrix(np.dot(np.diag(y), np.dot(K, np.diag(y))), (n, n), 'd')
    #     q = cvxopt.matrix(np.ones(n) * (-1), (n, 1), 'd')

    #     A = cvxopt.matrix(y, (1, n), "d")

    #     b = cvxopt.matrix(0.0)
    #     alpha = y * np.array(cvxopt.solvers.qp(P, q, G, h, A=A, b=b)["x"]).reshape(-1)

    #     support_vectors = np.where(np.abs(alpha) > tol)[0]
    #     intercept = 0
    #     for sv in support_vectors:
    #         intercept += y[sv]
    #         intercept -= np.sum(
    #             alpha[support_vectors] * K[sv, support_vectors])
    #     if len(support_vectors) > 0:
    #         intercept /= len(support_vectors)

    #     # set to zero non support vectors
    #     alpha[np.where(np.abs(alpha) <= tol)[0]] = 0
        
    #     self.intercept = intercept
    #     self.sv_indices = support_vectors
    #     self.alpha = alpha[support_vectors]
    #     self.sv_label = y[support_vectors]
    #     return

    def predict(self, K, norms_tr, norms_ts):
        """
        K has shape (N, n) where N are the train samples
        (among which we find the support vectors) and n
        are the test samples.
        """
        K_norm = K / np.outer(norms_tr, norms_ts)
        print(K_norm)
        K_norm_sv = K_norm[self.sv_indices, :]
        y_pred = np.dot(K_norm_sv.T, self.alpha * self.sv_label) + self.intercept
        return np.sign(y_pred)