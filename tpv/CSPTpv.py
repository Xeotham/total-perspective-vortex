import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin

I_TO_LABEL = {
    1: "T1",
    2: "T2",
}

LABEL_TO_I = {
    "T1": 1,
    "T2": 2,
}

"""
R1 and R2 are matrix corresponding to the spatial covariance.

The goal is to find a Weight matrix solving the CSP equation, which is:

D1 + D2 = Ic
Where Ic is a identity matrix (matrix with only the diagonal at 1 and the rest a 0)
of shape C * C, where C is the number of channels.

D1 and D2 are matrix corresponding to the operation:
D1 = Wt x R1 x W
D2 = Wt x R2 x W

Where W is the Weight matrix, Wt is it's transpose and R1/2 mean the Spatial Covariance of either one or two
and x is the Matrix Multiplication.

The spatial covariance use these equation:

R1 = mean(X1 x X1t / trace(X1 x X1t))
R2 = mean(X2 x X2t / trace(X2 x X2t))

Where X is the dataset, X1 and X2 are the part corresponding to the first and second label, Xt is the transpose of the
matrix, and trace is the sum of the diagonal components.

The decomposition of the equation is:

D1 + D2 = Ic
(Wt x R1 x W) + (Wt x R2 x W) = Ic
Wt x (R1 + R2) x W = Ic
"""

class CSPTpv(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 4):
        self.n_components = n_components

    @staticmethod
    def __calculate_covariance_matrix(X: np.ndarray) -> np.ndarray:
        covs = np.matmul(X, X.transpose((0, 2, 1)))
        trace = np.trace(covs, axis1=1, axis2=2)[:, None, None]

        covs_norm = covs / trace
        return np.mean(covs_norm, axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        values = np.unique(y)

        if values.shape[0] != 2:
            raise ValueError("The numbers of labels should be equal to 2")

        t1_list = X[y == values[0], :, :]
        t2_list = X[y == values[1], :, :]

        r1 = CSPTpv.__calculate_covariance_matrix(t1_list)
        r2 = CSPTpv.__calculate_covariance_matrix(t2_list)

        evals, W = scipy.linalg.eigh(r1, r1 + r2)

        # evals is the list of the channels from the one with the worst variance to the one
        # with the best (Ascending order), but we need to extraxt them the other way, the best one first, the worst one
        # in a second time. So we reverse the order of the weight.
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        W = W[:, idx]

        # MNE CSP normalize it's values, so we'll do the same
        for i in range(W.shape[1]):
            W[:, i] /= np.sqrt(W[:, i].T @ (r1 + r2) @ W[:, i])

        half_n = self.n_components // 2
        self.filters_ = np.hstack((W[:, :half_n], W[:, -half_n:]))
        return self

    def transform(self, X: np.ndarray, y: np.ndarray|None = None) -> np.ndarray:

        X_filtered = np.matmul(self.filters_.T, X)

        X_var = np.var(X_filtered, axis=-1)
        X_var /= np.sum(X_var, axis=-1, keepdims=True)

        return np.log(X_var)