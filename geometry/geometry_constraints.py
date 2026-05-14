import numpy as np
from numpy.linalg import solve
################# constraints ##############
def f_to_x(f):
    f = f.copy()
    N = len(f)
    x = np.zeros(N)
    f[0] = f[N - 1] - 2 * np.pi
    for k in range(1, N - 1):
        x[k] = np.log((f[k] - f[k - 1])/(f[k+1] - f[k]))
    x[N - 1] = (1/(np.pi - f[N - 1])) - (1/(np.pi + f[N - 1]))
    return x

def x_to_f(x, f0=None):
    N = len(x)
    f = np.zeros(N)

    # Invert x[N-1]: solve x*f^2 + 2f - x*pi^2 = 0, pick the right root
    xn = x[N-1]
    disc = 4 + 4 * xn**2 * np.pi**2
    f1 = (-2 + np.sqrt(disc)) / (2 * xn)
    f2 = (-2 - np.sqrt(disc)) / (2 * xn)
    def resid(fval):
        return abs((1/(np.pi - fval)) - (1/(np.pi + fval)) - xn)
    f[N-1] = f1 if resid(f1) < resid(f2) else f2

    # Reconstruct the boundary f_to_x used internally
    f[0] = f[N-1] - 2 * np.pi

    # Invert x[k] = log((f[k]-f[k-1])/(f[k+1]-f[k])) for k=1..N-2
    # => f[k]*(1+r) - f[k-1] - r*f[k+1] = 0,  r = exp(x[k])
    # Tridiagonal system with f[0] and f[N-1] as known boundaries
    M = N - 2
    r = np.exp(x[1:N-1])
    A = np.zeros((M, M))
    b = np.zeros(M)
    for i in range(M):
        A[i, i] = 1 + r[i]
        if i > 0:     A[i, i-1] = -1
        if i < M-1:   A[i, i+1] = -r[i]
        if i == 0:    b[i] += f[0]
        if i == M-1:  b[i] += r[i] * f[N-1]
    f[1:N-1] = solve(A, b)

    # f[0] is not recoverable from x; use provided value or leave as modified boundary
    if f0 is not None:
        f[0] = f0

    return f
