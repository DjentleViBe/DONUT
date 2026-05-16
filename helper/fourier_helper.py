import numpy as np

def fourier_encode(signal, K):
    N = len(signal)
    x = np.arange(N)

    a0 = np.mean(signal)

    ak = []
    bk = []

    for k in range(1, K + 1):
        ak.append(2/N * np.sum(signal * np.cos(2*np.pi*k*x/N)))
        bk.append(2/N * np.sum(signal * np.sin(2*np.pi*k*x/N)))

    return np.array([a0, *ak, *bk])


def fourier_decode(coeffs, N, K):
    a0 = coeffs[0]
    ak = coeffs[1:K+1]
    bk = coeffs[K+1:]

    x = np.arange(N)

    signal = np.full(N, a0)

    for k in range(1, K + 1):
        signal += ak[k-1] * np.cos(2*np.pi*k*x/N)
        signal += bk[k-1] * np.sin(2*np.pi*k*x/N)

    return signal