import numpy as np
import matplotlib.pyplot as plt

from Exercise7 import *



def reversed_enumerate(iter):
    return zip(range(len(iter)-1, -1, -1), reversed(iter))

if __name__ == "__main__":
        
    npoints = 10
    gammas = np.logspace(-2, -1, npoints)

    S = np.loadtxt("data/sp500.txt")
    n = S.shape[0]
    X0 = np.eye(n)

    for i, gamma in reversed_enumerate(gammas):
        print(f"optimizing for gamma={gamma}")
        X = accelerated_proximal_gradient(
            ft.partial(g, S=S),
            ft.partial(grad_g, S=S),
            ft.partial(prox_th, gamma=gamma),
            ft.partial(dual_gap, S=S, gamma=gamma),
            X0,
        )
        np.save(f"X_{i}.npy", X)
        X0 = X



