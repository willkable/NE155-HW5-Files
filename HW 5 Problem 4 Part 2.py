import numpy as np
from scipy.linalg import norm

# This is very similar to my previous code for problem 4; however, I have just replaced all the parameters with the ones
# given in the problem. All you should have to do is run the code and you'll get solution vectors,
# number of iterations, and the first absolute error below the tolerance

""""Makes Matrix"""""


def Matrix():
    n = 5
    x = 0
    tol = -6
    a = [-1] * (n - abs(-1))
    b = [4] * (n - abs(0))
    c = [-1] * (n - abs(1))
    A = np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)
    B = []
    if x == 0:
        x = np.zeros(len(A))
    for i in range(len(A)):
        B.append(100)
    return A, B, x, tol


"""Solves Using Jacobi Method"""


def Jacobi():
    print("Jacobi Iteration Method")
    A, B, x, tol = Matrix()
    Y = np.diag(A)
    Z = A - np.diagflat(Y)
    Max_Iters = 1000
    iters = 0
    for i in range(Max_Iters):
        iters += 1
        x_prev = x
        x = (B - np.dot(Z, x)) / Y
        error = np.linalg.norm((np.dot(A,x)-B))
        if error < (10 ** tol):
            print("Solution Vector: ", x)
            print("Number of Iterations: ", iters)
            print("Absolute Error: ", error)
            return ""


"""Solves Using Gauss-Seidel Method"""


def GS():
    print("Gauss-Seidel Iteration Method")
    A, B, x, tol = Matrix()
    Y = np.tril(A)
    Z = A - Y
    Max_Iter = 1000
    iters = 0
    for i in range(Max_Iter):
        iters += 1
        x_prev = x
        x = np.dot(np.linalg.inv(Y), B - np.dot(Z, x))
        error = np.linalg.norm((np.dot(A,x)-B))
        if error < (10 ** tol):
            print("Solution Vector: ", x)
            print("Number of Iterations: ", iters)
            print("Absolute Error: ", error)
            return ""


"""Solves Using SOR Method"""


def SOR():
    print("SOR Iteration Method")
    A, B, x, tol = Matrix()
    w = 1.1
    Upper = np.triu(A,1)
    Lower = np.tril(A,-1)
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
        error = np.linalg.norm((np.dot(A,x)-B))
        if error < 10**tol:
            print("Solution Vector: ", x)
            print("Number of Iterations: ", iters)
            print("Absolute Error: ", error)
            return ""


print(Jacobi())
print(GS())
print(SOR())