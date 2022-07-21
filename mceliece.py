import numpy as np
from aux_functions import GF2
from goppa import Goppa

class McEliece:
    
    def __init__(self, m, n, t):
        self.m = m
        self.n = n
        self.t = t
        self.goppaCode = None
        self.S = None
        self.P = None
        self.SGP = None
        
    def keyGen(self):
        self.goppaCode = Goppa(self.m, self.n, self.t)
        G, _ = self.goppaCode.generate()
        
        # Matriz S invertible aleatoria de dimension k
        self.S = GF2.Random((self.goppaCode.k, self.goppaCode.k))
        while np.linalg.det(self.S) == 0:
             self.S = GF2.Random((self.goppaCode.k, self.goppaCode.k))
           
        # Matriz de permutacion P aleatoria de dimension n
        self.P = GF2(np.random.permutation(np.eye(self.n, dtype=int)))
        
        self.SGP = (self.S).dot(G).dot(self.P)
        return (self.SGP, self.t), (self.S, G, self.P)
    
    def encrypt(self, message):
        if len(message) != self.SGP.shape[0]: #self.goppaCode.k
            print(f"McEliece.encrypt: El mensaje tiene que ser de longitud {self.goppaCode.k}")
            return
        
        # Generar vector de longitud n con t errores, y barajarlo
        error = np.zeros(self.n, dtype=int)
        error[0:self.t] = 1
        np.random.shuffle(error)
        
        # Codificar con el codigo Goppa de matriz generadora SGP y añadir errores
        #ciphertext = np.array(message).dot(np.array(self.SGP)) + np.array(error)
        ciphertext = GF2(message).dot(self.SGP) + GF2(error)
        return ciphertext, error
    
    def decrypt(self, ciphertext):
        if len(ciphertext) != self.n:
            print(f"McEliece.decrypt: El texto cifrado tiene que ser de longitud {self.n}")
            return
        
        # m * SGP * P^-1 + e * P^-1
        invP = np.linalg.inv(self.P)
        codeword = ciphertext.dot(invP)

        # m * S 
        mS, errorVector = self.goppaCode.decode(codeword)

        # m * S * S^-1
        invS = np.linalg.inv(self.S)
        m = mS.dot(invS)
        return m, errorVector
    