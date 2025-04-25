import functools as ft
import numpy as np
import matplotlib.pyplot as plt

def g(X, S):
    return np.sum(X*S) - np.linalg.slogdet(X)[1]

def grad_g(X, S):
    return S - np.linalg.inv(X)

def prox_th(X, t, gamma):
    A = np.sign(X) * np.maximum(np.abs(X)-gamma*t,0)
    np.fill_diagonal(A, np.diag(X))
    return A

def proximal_gradient(g, grad_g, prox_th, x0, beta):
    x = x0
    for i in range(1000):
        t = 1e-1
        grad_gx = grad_g(x)
        gx = g(x)
        while True:
            x_tent = prox_th(x-t*grad_gx, t)
            step = x_tent - x
            if g(x_tent) <= gx + np.sum(grad_gx*step) + 1/(2*t)*np.sum(step*step):
                break
            t = t*beta
        x = x_tent
    return x

X0 = np.eye(4)
# A = np.random.rand(5, 5)
# S = np.dot(A, A.T)
S = np.array([
    [1.00, -0.50, 0.10, -0.90],
    [-0.50, 1.25, -0.05, 1.05],
    [0.10, -0.05, 0.26, -0.09],
    [-0.90, 1.05, -0.09, 5.17]
])
gamma = 0.01

X = proximal_gradient(ft.partial(g,S=S), ft.partial(grad_g,S=S), ft.partial(prox_th,gamma=gamma), X0, beta=0.8)
print(X)
print(np.linalg.inv(X)-S)