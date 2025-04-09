import cvxpy as cp
import numpy as np

def cvxSolve(S, gamma):
    X = cp.Variable(S.shape)
    objective = cp.Minimize(cp.trace(S@X) - cp.log_det(X) + gamma*(cp.norm1(X) - cp.sum(cp.diag(X))))
    prob = cp.Problem(objective, [])

    result = prob.solve()
    print(X.value)
    print(result)
    print(np.linalg.inv(X.value) - S)

n = 4 
A = np.random.rand(n, n)
S = np.dot(A, A.T)
gamma = 0.3

print(S)
cvxSolve(S, gamma)