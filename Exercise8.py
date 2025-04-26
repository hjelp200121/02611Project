import numpy as np
import matplotlib.pyplot as plt

from Exercise7 import *

def off_diag_nnz(X):
    X = np.copy(X)
    np.fill_diagonal(X, 1)
    return np.sum(np.abs(X) > 1e-2)

def reversed_enumerate(iter):
    return zip(range(len(iter)-1, -1, -1), reversed(iter))

if __name__ == "__main__":
        
    npoints = 10
    gammas = np.logspace(-2, -1, npoints)

    S = np.loadtxt("data/sp500.txt")
    n = S.shape[0]
    X0 = np.load("X.npy")

    gs = np.zeros(npoints)
    hs = np.zeros(npoints)
    nnz = np.zeros(npoints)

    for i, gamma in reversed_enumerate(gammas):
        print(f"optimizing for gamma={gamma}")
        X = proximal_gradient(
            ft.partial(g, S=S),
            ft.partial(grad_g, S=S),
            ft.partial(prox_th, gamma=gamma),
            ft.partial(dual_gap, S=S, gamma=gamma),
            X0,
        )
        X0 = X
        gs[i] = g(X, S)
        hs[i] = h(X, gamma)
        nnz[i] = off_diag_nnz(X)
        
    np.save("g.npy", gs)
    np.save("h.npy", hs)
    np.save("nnz.npy", nnz)


