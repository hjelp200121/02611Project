import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def cvxSolve(S, gamma):
    X = cp.Variable(S.shape)
    objective = cp.Minimize(cp.trace(S@X) - cp.log_det(X) + gamma*(cp.norm1(X) - cp.sum(cp.diag(X))))
    prob = cp.Problem(objective, [])

    result = prob.solve()

    return result, X.value

npoints = 100
gammas = np.logspace(-2, 0, npoints)
S = np.array([
    [1.00, -0.50, 0.10, -0.90],
    [-0.50, 1.25, -0.05, 1.05],
    [0.10, -0.05, 0.26, -0.09],
    [-0.90, 1.05, -0.09, 5.17]
])
x = np.zeros(npoints)
y = np.zeros(npoints)

for i, gamma in enumerate(gammas):
    result, X = cvxSolve(S, gamma)
    if i == 0:
        print(gamma)
        print(X)
        print(np.linalg.inv(X)-S)
    x[i] = np.trace(S@X) - np.linalg.slogdet(X)[1]
    y[i] = np.sum(np.abs(X)) - np.sum(np.diag(X))

fig = plt.figure()
ax = fig.subplots()
ax.plot(x,y)
ax.set_xlabel("$f_1(X(\\gamma))$")
ax.set_ylabel("$f_2(X(\\gamma))$")
fig.tight_layout()
fig.savefig("plot.pdf")