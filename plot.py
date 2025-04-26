import numpy as np
import matplotlib.pyplot as plt

from Exercise7 import g, h

def off_diag_nnz(X):
    X = np.copy(X)
    np.fill_diagonal(X, 1.0)
    return np.sum(np.abs(X) > 1e-4)

if __name__ == "__main__":

    npoints = 10
    S = np.loadtxt("data/sp500.txt")
    gammas = np.logspace(-2, -1, npoints)
    gammas = gammas[1:]

    Xs = [np.load(f"X_{i}.npy") for i in range(1, npoints)]
    gs = [g(X, S) for X in Xs]
    hs = [h(X, 1.0) for X in Xs]
    nnzs = [off_diag_nnz(X) for X in Xs]

    # trade off curve
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(gs, hs)
    ax.set_xlabel("$g(X^\\star(\\gamma))$")
    ax.set_ylabel("$h(X^\\star(\\gamma))$")
    fig.tight_layout()
    fig.savefig("tradeoff.pdf")


    # nnz
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(gammas, nnzs)
    ax.set_xlabel("$\\gamma$")
    ax.set_ylabel("Off-diagonal non-zeros")
    fig.tight_layout()
    fig.savefig("nnz.pdf")



