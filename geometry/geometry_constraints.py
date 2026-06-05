"""
Constraint equations
"""
import numpy as np
from numpy.linalg import solve
################# constraints ##############
def f_to_x(f):
    """
    function to X values
    """
    f = f.copy()
    nval = len(f)
    x = np.zeros(nval)
    f[0] = f[nval - 1] - 2 * np.pi
    for k in range(1, nval - 1):
        x[k] = np.log((f[k] - f[k - 1])/(f[k+1] - f[k]))
    x[nval - 1] = (1/(np.pi - f[nval - 1])) - (1/(np.pi + f[nval - 1]))
    return x

def x_to_f(x, f0=None):
    """
    X values to function
    """
    nval = len(x)
    f = np.zeros(nval)

    # Invert x[N-1]: solve x*f^2 + 2f - x*pi^2 = 0, pick the right root
    xn = x[nval-1]
    disc = 4 + 4 * xn**2 * np.pi**2
    f1 = (-2 + np.sqrt(disc)) / (2 * xn)
    f2 = (-2 - np.sqrt(disc)) / (2 * xn)
    def resid(fval):
        return abs((1/(np.pi - fval)) - (1/(np.pi + fval)) - xn)
    f[nval-1] = f1 if resid(f1) < resid(f2) else f2

    # Reconstruct the boundary f_to_x used internally
    f[0] = f[nval-1] - 2 * np.pi

    # Invert x[k] = log((f[k]-f[k-1])/(f[k+1]-f[k])) for k=1..N-2
    # => f[k]*(1+r) - f[k-1] - r*f[k+1] = 0,  r = exp(x[k])
    # Tridiagonal system with f[0] and f[N-1] as known boundaries
    mval = nval - 2
    rval = np.exp(x[1:nval-1])
    aval = np.zeros((mval, mval))
    bval = np.zeros(mval)
    for i in range(mval):
        aval[i, i] = 1 + rval[i]
        if i > 0:
            aval[i, i-1] = -1
        if i < mval-1:
            aval[i, i+1] = -rval[i]
        if i == 0:
            bval[i] += f[0]
        if i == mval-1:
            bval[i] += rval[i] * f[nval-1]
    f[1:nval-1] = solve(aval, bval)

    # f[0] is not recoverable from x; use provided value or leave as modified boundary
    if f0 is not None:
        f[0] = f0

    return f

def f_to_u(f, eps=1e-12, period=2*np.pi):
    """Convert function to periodic data
    """
    f = np.asarray(f)
    df = np.diff(f)
    wrap = (f[0] + period) - f[-1]
    df = np.append(df, wrap)
    return np.log(df + eps)

def u_to_f(u, f0=0.0, period=2*np.pi):
    """
    Convert periodic data to function
    """
    u = np.asarray(u, dtype=float)

    # 1. recover raw segment lengths
    df = np.exp(u)
    # 2. enforce circular closure constraint
    df *= period / np.sum(df)
    # 3. integrate
    f = np.zeros(len(u) + 1)
    f[0] = f0
    for k in range(len(u)):
        f[k+1] = f[k] + df[k]
    return f[:-1]
