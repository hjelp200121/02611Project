import functools as ft
import numpy as np
import matplotlib.pyplot as plt

def is_spd(x):
    try:
        np.linalg.cholesky(x)
        return True
    except np.linalg.LinAlgError:
        return False

def g(X, S):
    return np.sum(X*S) - np.linalg.slogdet(X)[1]

def h(X, gamma):
    return gamma*(np.sum(np.abs(X)) - np.sum(np.diag(X)))

def grad_g(X, S):
    return S - np.linalg.inv(X)

def prox_th(X, t, gamma):
    A = np.sign(X) * np.maximum(np.abs(X)-gamma*t,0)
    np.fill_diagonal(A, np.diag(X))
    return A

def compute_dual(X, gamma, S):
    X_inv_S = np.linalg.inv(X) - S
    U = np.clip(X_inv_S, -gamma, gamma)
    np.fill_diagonal(U, 0)
    return U


def dual_gap(X, S, gamma):
    U = compute_dual(X, gamma, S)

    if not is_spd(S + U):
        return np.inf

    _, logdet = np.linalg.slogdet(S + U)
    return g(X, S) + h(X, gamma) - logdet - S.shape[0]


def proximal_gradient(g, grad_g, prox_th, dual_gap, x0, beta=0.5, atol=1e-2, max_iter=10000):
    x = x0
    
    for i in range(max_iter):
        t = 1.0
        grad_gx = grad_g(x)
        gx = g(x)
        while True:
            x_tent = prox_th(x-t*grad_gx, t)
            step = x_tent - x
            gx_tent = g(x_tent)

            cond1 = is_spd(x_tent)
            cond2 = gx_tent <= gx + np.sum(grad_gx*step) + 1/(2*t)*np.sum(step*step)

            if cond1 and cond2:
                break

            t = t*beta
        
        x = x_tent
        gx = gx_tent
        delta = dual_gap(x)

        if delta < atol:
            break

        if i % 100 == 0:
            print(f"iteration: {i}, dual gap: {delta}")

    return x

def armijo(gx, grad_gx, gx_tent, step, t):
    return gx_tent <= gx + np.sum(grad_gx*step) + 1/(2*t)*np.sum(step*step)

def step(x, g, grad_g, prox_th, beta):
    t = 1.0
    grad_gx = grad_g(x)
    gx = g(x)
    while True:
        x_tent = prox_th(x - t*grad_gx, t)
        step = x_tent - x

        if armijo(gx, grad_gx, g(x_tent), step, t) and is_spd(x_tent):
            break

        t = t*beta
    
    return x_tent, t

def accelerated_proximal_gradient(g, grad_g, prox_th, dual_gap, x0, beta=0.9, atol=1e-2, max_iter=10000):
    x = x0
    y = x
    gamma = 1.0

    for i in range(max_iter):
        delta = dual_gap(x)
        if i % 100 == 0:
            print(f"iteration: {i}, dual gap: {delta}")
        if delta < atol:
            break

        x_next, _t = step(y, g, grad_g, prox_th, beta)
        gamma_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * gamma * gamma))
        y = x_next + ((gamma - 1.0) / gamma_next) * (x_next - x)
        x = x_next
        gamma = gamma_next

    return x


if __name__ == "__main__":
    # A = np.random.rand(5, 5)
    # S = np.dot(A, A.T)
    # S = np.array([
    #     [1.00, -0.50, 0.10, -0.90],
    #     [-0.50, 1.25, -0.05, 1.05],
    #     [0.10, -0.05, 0.26, -0.09],
    #     [-0.90, 1.05, -0.09, 5.17]
    # ])
    S = np.loadtxt("data/sp500.txt")
    X0 = np.load("X.npy")
    gammas = np.logspace(-2, -1, 10)[-2::-1]
    
    for gamma in gammas:
        print(f"gamma: {gamma}")
        X = proximal_gradient(
            ft.partial(g, S=S),
            ft.partial(grad_g, S=S),
            ft.partial(prox_th, gamma=gamma),
            ft.partial(dual_gap, S=S, gamma=gamma),
            X0,
        )
