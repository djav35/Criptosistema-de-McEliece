import galois
    
''' Cuerpo GF(2) '''
GF2 = galois.GF(2)

''' Algoritmo extendido de Euclides con polinomios sobre GF '''
def extendedEuclidPoly(p, q, GF):
    if q.degree > p.degree:
        (x, y, d) = extendedEuclidPoly(q, p, GF)
        return (y, x, d)

    x1, x2, y1, y2 = galois.Poly.Zero(GF), galois.Poly.One(GF), galois.Poly.One(GF), galois.Poly.Zero(GF)
    
    while q != 0:
        c, r = divmod(p, q)
        x = x2 - c * x1
        y = y2 - c * y1
        p, q, x2, x1, y2, y1 = q, r, x1, x, y1, y

    return (x2, y2, p)

''' Modificacion de extendedEuclidPoly para adaptarlo al algoritmo de Patterson '''
def extendedEuclidPolyMod(a, b, t, GF):
    if b.degree > a.degree:
        (x, y, d) = extendedEuclidPolyMod(b, a, t, GF)
        return (y, x, d)

    x1, x2, y1, y2 = galois.Poly.Zero(GF), galois.Poly.One(GF), galois.Poly.One(GF), galois.Poly.Zero(GF)
    
    while not a.degree <= t // 2 or not x2.degree <= (t - 1) // 2:
        q, r = divmod(a, b)
        x = x2 - q * x1
        y = y2 - q * y1
        a, b, x2, x1, y2, y1 = b, r, x1, x, y1, y

    return (x2, y2, a)

''' Inverso modular de un polinomio '''
def inv(p, g, GF):
    a, b, c = extendedEuclidPoly(p, g, GF)
    return a / c

''' 
Separar un polinomio en potencias pares e impares 
A^2 + zB^2 -> A, B
'''
def split(p, GF):
    coeffs1 = [squareRoot(coeff, GF) for coeff in p.coeffs[0::2]]
    coeffs2 = [squareRoot(coeff, GF) for coeff in p.coeffs[1::2]]
    a = galois.Poly(coeffs1, field=GF)
    b = galois.Poly(coeffs2, field=GF)
    
    # Devolver primero el de grado par
    if p.degree % 2 == 0:
        return a, b
    else:
        return b, a

''' Raiz cuadrada de un elemento de GF(2^m): a^(2^(m-1)) '''
def squareRoot(elem, GF):
    return elem**(2**(GF.degree - 1))
