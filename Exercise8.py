import numpy as np
import matplotlib.pyplot as plt

from Exercise7 import *

def number_of_off_diag_zeros(X):
    np.fill_diagonal(X, 1)
    return np.sum(np.abs(X) > 1e-2)

npoints = 10
gammas = np.logspace(-2, -1, npoints)

S = np.loadtxt("data/sp500.txt")
n = S.shape[0]
X0 = np.eye(n)

x = np.zeros(npoints)
y = np.zeros(npoints)
off_diag_zeros = np.zeros(npoints)

for i, gamma in enumerate(gammas):
    X = proximal_gradient(ft.partial(g,S=S), ft.partial(grad_g,S=S), ft.partial(prox_th,gamma=gamma), X0, beta=0.8)

    x[i] = np.trace(S@X) - np.linalg.slogdet(X)[1]
    y[i] = np.sum(np.abs(X)) - np.sum(np.diag(X))
    off_diag_zeros[i] = number_of_off_diag_zeros(X)

# plot trade off curve
plt.plot(x,y)
plt.savefig("test2.pdf")

# clear plot
plt.cla()

# plot off diagonal zeros
plt.plot(gammas, off_diag_zeros)
plt.savefig("plot2.pdf")
