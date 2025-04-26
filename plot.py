import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    npoints = 10
    gammas = np.logspace(-2, -1, npoints)

    gs = np.load("g.npy")
    hs = np.load("h.npy")
    nnz = np.load("nnz.npy")

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
    ax.plot(gammas, nnz)
    ax.set_xlabel("$\\gamma$")
    ax.set_ylabel("Off-diagonal non-zeros")
    fig.tight_layout()
    fig.savefig("nnz.pdf")



