import numpy as np
from scipy.linalg import norm

# All you should have to do is run the code


""""Makes Matrix"""""


def Matrix():
    n = 5
    x = 0
    a = [-1] * (n - abs(-1))
    b = [4] * (n - abs(0))
    c = [-1] * (n - abs(1))
    A = np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)
    B = []
    if x == 0:
        x = np.zeros(len(A))
    for i in range(len(A)):
        B.append(100)
    return A, B, x


"""Solves Using Jacobi Method"""


def Jacobi(tol):
    print("Jacobi Iteration Method")
    A, B, x = Matrix()
    Y = np.diag(A)
    Z = A - np.diagflat(Y)
    Max_Iters = 1000
    iters = 0
    for i in range(Max_Iters):
        iters += 1
        x_prev = x
        x = (B - np.dot(Z, x)) / Y
        error = norm(x - x_prev) / norm(x)
        if error < tol:
            print("Solution Vector: ", x)
            print("Number of Iterations: ", iters)
            print("Relative Error: ", error)
            return ""


"""Solves Using Gauss-Seidel Method"""


def GS(tol):
    print("Gauss-Seidel Iteration Method")
    A, B, x = Matrix()
    Y = np.tril(A)
    Z = A - Y
    Max_Iter = 1000
    iters = 0
    for i in range(Max_Iter):
        iters += 1
        x_prev = x
        x = np.dot(np.linalg.inv(Y), B - np.dot(Z, x))
        error = norm(x - x_prev) / norm(x)
        if error < tol:
            print("Solution Vector: ", x)
            print("Number of Iterations: ", iters)
            print("Relative Error: ", error)
            return ""


"""Solves Using SOR Method"""


def SOR(tol):
    print("SOR Iteration Method")
    A, B, x = Matrix()
    w = 1.1
    Upper = np.triu(A, 1)
    Lower = np.tril(A, -1)
    Middle = np.diagflat(np.diag(A))
    DiagInverse = np.linalg.inv(Middle + Lower * w)
    iters = 0
    Max_Iters = 1000
    wB = []
    wU = []
    for i in B:
        wB.append(w * i)
    for i in Upper:
        wU.append(w * i)
    x = np.matrix.transpose(x)
    for i in range(Max_Iters):
        iters += 1
        x_prev = x
        x = np.dot(DiagInverse, wB - np.dot((wU + (w - 1) * Middle), x))
        error = np.linalg.norm(x - x_prev) / np.linalg.norm(x)
        if error < tol:
            print("Solution Vector: ", x)
            print("Number of Iterations: ", iters)
            print("Relative Error: ", error)
            return ""


print("Tolerance at 10^-6")
print(Jacobi(10 ** -6))
print(GS(10 ** -6))
print(SOR(10 ** -6))

print("Tolerance at 10^-8")
print(Jacobi(10**-8))
print(GS(10**-8))
print(SOR(10**-8))

