import numpy as np

#Cria uma sequência de 2^n números, no intervalo de [1, 1000) 
def create_sequence(n):
    return np.random.rand(2**n) * 1000


#Calcula a DFT de uma sequência de números
def DFT(seq):

    if len(seq) == 1:
        return seq
    else:
        #Computa recursivamente as sequências pares e ímpares
        N = len(seq)
        G = DFT(seq[::2]) #Seleciona os termos pares
        H = DFT(seq[1::2]) #Seleciona os termos ímpares
        W = np.exp(-2j*np.pi*np.arange(0, N/2)/N)
        WH = W*H

        return np.concatenate((G + WH, G - WH))



if __name__ == "__main__":

    print(DFT([1, -1, 1, -1, 5, 4, 3, 2]))


