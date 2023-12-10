#!/usr/bin/env python
""" 
Implementa a operação de DFT sobre uma sequência de números
"""

__author__ = "Lucas Carvalho and Rafael Marasca Martins"

import numpy as np
from mpi4py import MPI

def create_sequence(n):
    return np.random.rand(n) * 1000


#Calcula K valores da DFT para a sequência x
def dft(x, K):

    N = len(x)
    X = np.zeros(len(K), dtype = 'complex_')

    for i,k in enumerate(K):
        for n in range(0,N):
            X[i] += x[n] * np.exp(-2j*np.pi*n*k/N)

    return X


#Calcula a FFT da sequência
def fft(x):

    if len(x) == 1:
        return x
    else:
        #Computa recursivamente as sequências pares e ímpares
        N = len(x)

        G = fft(x[::2]) #Seleciona os termos pares
        H = fft(x[1::2]) #Seleciona os termos ímpares

        #Gera os vetores de exponencial
        W = np.exp(-2j*np.pi*np.arange(0, N/2)/N)
        WH = W*H

        return np.concatenate((G + WH, G - WH)) #Concatena e retorna os resultados


if __name__ == "__main__":

    #Inicializa o comunicador do MPI
    comm = MPI.COMM_WORLD

    #Número do processo
    rank = comm.Get_rank()

    #Número de processos
    n_proc = comm.Get_size()

    #Tamanho da sequência
    N = 4096
    
    #Inicializa os valores de x, K e data_size
    if rank == 0:
        x = create_sequence(N)
        K = np.arange(0,N,dtype = np.int32)
        data_size = N//n_proc
        t1 = MPI.Wtime()
    else:
        data_size = None
        x = None
        K = None

    #Emite o tamanho dos vetores para todos os processos
    data_size = comm.bcast(data_size , 0)

    #Emite a sequência de números para todos os processos
    x = comm.bcast(x, 0)

    #Inicializa os valores de K_temp
    K_temp = np.zeros(data_size, dtype = np.int32)

    #Espalha os valores de K entre os processos
    comm.Scatter(K, K_temp, 0) 

    #Calcula os valores da DFT em K_temp
    X_temp = dft(x, K_temp)

    #Agrega todos os resultados
    X = comm.gather(X_temp, root = 0)

    #Resultados
    if rank == 0:
        t2 = MPI.Wtime()
        print("Versão Paralela: ")
        print(np.concatenate(X))
        print(t2 - t1)
        print("-----------------------------------")

        t1 = MPI.Wtime()
        X = dft(x, K)
        t2 = MPI.Wtime()
        print("Versão Serial:")
        print(X)
        print(t2 - t1)
        print("-----------------------------------")

        t1 = MPI.Wtime()
        X = fft(x)
        t2 = MPI.Wtime()
        print("FFT:")
        print(X)
        print(t2 - t1)
        print("-----------------------------------")
