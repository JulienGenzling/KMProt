import cvxopt
import numpy as np

cvxopt.solvers.options["show_progress"] = False


class Fitter:
    def __init__(self):
        self.intercept = None
        self.alpha = None

    def fit(self, K, y):
        """
        N_tr = number of training samples
        N_ts = number of test samples
        K = np.array of shape (N_tr, N_tr)
        y = np.array of shape (N_tr,)
        """
        raise NotImplementedError

    def predict(self, K, norms_tr, norms_ts):
        """
        K = np.array of shape (N_tr, N_ts)
        norms_tr = np.array of shape (N_tr,)
        norms_ts = np.array of shape (N_ts,)
        """
        raise NotImplementedError


class WeightedKRR(Fitter):
    def __init__(self, **params):
        """
        Weighted Kernel Ridge Regression for binary classification.
    
        Solves the optimization problem:
        min_f ∑w_i(y_i - f(x_i))² + λ||f||²
        
        Solution: α = W^(1/2)(W^(1/2)KW^(1/2) + λnI)⁻¹W^(1/2)y
        where W is the diagonal matrix of weights, K is the kernel matrix,
        λ is the regularization parameter, and y are the labels.
        """
        super().__init__()
        self.params = {"C": 1.0}
        self.params.update(params)

    def fit(self, K, y, w=None):
        n = K.shape[0]
        C = self.params["C"]
        
        if w is None:
            w = np.ones(n)
        
        W_half = np.diag(np.sqrt(w))
        K_weighted = W_half @ K @ W_half
        inv_matrix = np.linalg.inv(K_weighted + n * C * np.eye(n))
        self.alpha = W_half @ inv_matrix @ W_half @ y
        
        y_pred = np.dot(K, self.alpha)
        self.intercept = np.mean(y - y_pred)
        
        return self.alpha

    def predict(self, K, norms_tr, norms_ts):
        K_norm = K / np.outer(norms_tr, norms_ts)
        y_pred = np.dot(K_norm.T, self.alpha) + self.intercept
        return np.sign(y_pred)


class KRR(Fitter):
    def __init__(self, **params):
        """
        Kernel Ridge Regression for binary classification.
    
        Solves the optimization problem:
        min_f ∑(y_i - f(x_i))² + λ||f||²
        
        Solution: α = (K + λnI)⁻¹y
        where K is the kernel matrix, λ is the regularization parameter,
        n is the number of samples, and y are the labels.
        """
        super().__init__()
        self.params = {"C": 1.0}
        self.params.update(params)
        self.weighted_krr = WeightedKRR(**self.params)

    def fit(self, K, y):
        self.alpha = self.weighted_krr.fit(K, y) # Use WeightedKRR with uniform weights
        self.intercept = self.weighted_krr.intercept
        return self.alpha

    def predict(self, K, norms_tr, norms_ts):
        return self.weighted_krr.predict(K, norms_tr, norms_ts)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class KLR(Fitter):
    def __init__(self, **params):
        """
        Kernel Logistic Regression for binary classification.
        
        Solves the optimization problem:
        min_f ∑log(1 + e^(-y_i·f(x_i))) + λ||f||²
        
        Solution: Uses Iteratively Reweighted Least Squares (IRLS) algorithm
        by solving a sequence of weighted KRR problems:
        1. Compute weights w_i = p_i(1-p_i) where p_i = sigmoid(y_i·f(x_i))
        2. Compute working targets z_i = f(x_i) + (y_i-p_i)/w_i
        3. Solve weighted KRR with weights w_i and targets z_i
        4. Repeat until convergence
        """
        super().__init__()
        self.params = {"C": 0.01, "n_iter": 10, "tol": 1e-6, "eps": 1e-6}
        self.params.update(params)

    def fit(self, K, y):
        n = K.shape[0]
        alpha_old = np.zeros(n)
        self.alpha = np.zeros(n)
        
        for i in range(self.params["n_iter"]):
            M = K @ self.alpha
            
            sig_pos = sigmoid(M * y)
            sig_neg = sigmoid(-M * y)
            W = sig_neg * sig_pos
            
            Z = M + y / np.maximum(sig_pos, self.params["eps"])
            
            alpha_old = self.alpha.copy()
            
            wkrr = WeightedKRR(C=self.params["C"])
            self.alpha = wkrr.fit(K, Z, w=W)
            
            if np.linalg.norm(self.alpha - alpha_old) < self.params["tol"]:
                print(f"KLR converged in {i+1} iterations")
                break
                
            if i == self.params["n_iter"] - 1:
                print("WARNING: KLR didn't converge")
        
        y_pred_prob = sigmoid(K @ self.alpha)
        y_pred = np.where(y_pred_prob > 0.5, 1, -1)
        self.intercept = np.mean(y - y_pred)
        
        return self.alpha

    def predict(self, K, norms_tr, norms_ts):
        K_norm = K / np.outer(norms_tr, norms_ts)
        raw_pred = K_norm.T @ self.alpha + self.intercept
        probs = sigmoid(raw_pred)
        return np.where(probs > 0.5, 1, -1)

class SVM(Fitter):
    def __init__(self, **params):
        """
        SVM for binary classification

        We solve the SVM dual problem :
        max sum alpha_i - 1/2 sum_{j,k} alpha_j alpha_k y_j y_k K(x_j, x_k)
            st. 0 <= alpha_i <= C and sum alpha_i * y_i = 0
        """
        super().__init__()
        self.params = {"C": 1.0, "tol": 1e-4}
        self.params.update(params)

    def fit(self, K, y):
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
        K_norm = K / np.outer(norms_tr, norms_ts)
        y_pred = (
            np.dot(K_norm[self.sv_indices, :].T, self.alpha * self.sv_label)
            + self.intercept
        )
        return np.sign(y_pred)

