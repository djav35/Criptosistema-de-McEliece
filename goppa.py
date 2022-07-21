import numpy as np
import sympy as sp
import galois
from aux_functions import GF2, inv, split, extendedEuclidPolyMod

class Goppa:
    
    def __init__(self, m, n, t):
        self.m = m
        self.n = n
        self.t = t
        self.q = 2
        self.k = None
        self.GF = None
        self.g = None
        self.L = None
        self.H = None
        self.binH = None
        self.G = None   
        
    def generate(self):  
        if (self.q**self.m < self.n):
            print(f"Goppa.generate: No se puede construir el código Goppa con estos parámetros: q={self.q}^m={self.m} < n={self.n}")
            return
        
        # Cuerpo GF(2^m)
        self.GF = galois.GF(self.q**self.m)
        
        # Generar g aleatorio, monico e irreducible (para evitar problemas) de grado t sobre GF 
        self.g = galois.irreducible_poly(self.q**self.m, self.t, method="random")
        
        # n elementos distintos y aleatorios de GF (que no son raices de g)
        self.L = self.GF(np.random.choice(self.GF.Elements(), self.n, replace=False))
        
        # Crear H a partir de las matrices auxiliares X, Y, Z sobre GF
        X = self.GF(np.zeros((self.t, self.t), dtype=int))
        for i in range(self.t):
            for j in range(self.t):
                if i - j >= 0:
                    X[i,j] = self.g.coeffs[i - j]           
        
        Y = self.GF(np.zeros((self.t, self.n), dtype=int))
        for i in range(self.t):
            for j in range(self.n):
                Y[i,j] = self.L[j]**i
          
        Z = self.GF(np.zeros((self.n, self.n), dtype=int))
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    Z[i,j] = self.g(self.L[i])**-1    

        self.H = X.dot(Y).dot(Z)
        
        # Pasar a vector cada fila de H y concatenarlas
        self.binH = np.concatenate(tuple(self.H[i].vector().T for i in range(len(self.H))))
        
        self.G = self.binH.null_space()
        self.k = self.G.shape[0]
        return self.G, self.binH
    
    def encode(self, message):
        if len(message) != self.k:
            print(f"Goppa.encode: El mensaje tiene que ser de longitud {self.k}")
            return
        codeword = GF2(message).dot(self.G)
        return codeword

    def decode(self, codeword):
        if len(codeword) != self.n:
            print(f"Goppa.decode: La palabra código tiene que ser de longitud {self.n}")
            return
        
        # Calcular el sindrome
        syndromeCoeffs = self.H.dot(self.GF(codeword))
        syndromePoly = galois.Poly(syndromeCoeffs, field=self.GF)
        
        # Polinomio localizador sigma = a^2 + zb^2
        # a = bR mod g
        H0, H1 = split(self.g, self.GF)
        w = H0 * inv(H1, self.g, self.GF)
        T = inv(syndromePoly, self.g, self.GF)
        T0, T1 = split(T + galois.Poly.Identity(self.GF) , self.GF)
        R = T0 + w * T1
        b, _, a = extendedEuclidPolyMod(R, self.g, self.t, self.GF)
        sigma = a**2 + galois.Poly.Identity(self.GF) * b**2
        
        # Ver las posiciones de los errores con las raices del polinomio localizador
        errorIndex = [list(self.L).index(root) for root in sigma.roots() if root in self.L]
        errorVector = np.zeros(self.n, dtype=int)
        errorVector[errorIndex] = 1
        
        # Corregir errores
        word = codeword - GF2(errorVector)
 
        # Resolver el sistema G^T * mS = word, a través de la forma RREF de G
        indexes = sp.Matrix(self.G).rref()[1]
        mS = word[np.array(indexes)]
        return mS, errorVector
