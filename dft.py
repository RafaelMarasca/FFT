import numpy as np


def create_sequence(n):
    return np.random.rand(2**n) * 1000


def dft(x):
    N = len(x)
    W = np.exp(-2j*np.pi*np.arange(0, N)/N)

    X = np.zeros(N,dtype = 'complex_')

    for k in range(0,N):
        for n in range(0,N):
            X[k] += x[n] * np.exp(-2j*np.pi*n*k/N)

    return X


if __name__ == "__main__":

    print(dft(np.array([1, -1, 1, -1, 5, 4, 3, 2])))