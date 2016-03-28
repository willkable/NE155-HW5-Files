import numpy as np
from Tools.scripts.treesync import raw_input
from scipy.linalg import norm

# Assuming that the given matrix does not change on the diagonal and the B vector does not change,
# then all that would have to be done is follow the prompts in the console and input the number of
# unknowns, your initial guess, and a tolerance (For SOR you will also need a w). For the tolerance all you have
# to do is input the number that you want 10**x. So if you want 10^-6, Just input -6 for tolerance.
# If your guess is just the 0 vector, then you can just input 0 for guess, otherwise input the whole
# guess vector in the form [x1, x2, x3, ...., xn]. The functions will return a solution vector
# [x1 x2 x3 ... xn] and the number of iterations to meet the given error tolerance. I have a max
# iteration set to 1000 and i used relative errors for all the iterations.

print("Press 1 for Jacobi Method")
print("Press 2 for Gauss-Seidel Method")
print("Press 3 for SOR Method")
T = int(raw_input("Choose Now: "))

""""Makes Matrix"""""


def Matrix():
    n = int(raw_input("Number of Unknowns: "))
    x = eval((raw_input("Initial Guess for Solution: ")))
    tol = int(raw_input("Tolerance: "))
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
        error = norm(x - x_prev) / norm(x)
        if error < (10 ** tol):
            print("Solution Vector: ", x)
            print("Number of Iterations: ", iters)
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
        error = norm(x - x_prev) / norm(x)
        if error < (10 ** tol):
            print("Solution Vector: ", x)
            print("Number of Iterations: ", iters)
            return ""


"""Solves Using SOR Method"""


def SOR():
    print("SOR Iteration Method")
    A, B, x, tol = Matrix()
    w = float(raw_input("Omega Values: "))
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
    for i in range(Max_Iters):
        iters += 1
        x_prev = x
        x = np.dot(DiagInverse, wB - np.dot((wU + (w - 1) * Middle), x))
        error = np.linalg.norm(x - x_prev) / np.linalg.norm(x)
        if error < 10**tol:
            print("Solution Vector: ", x)
            print("Number of Iterations: ", iters)
            return ""


if T == 1:
    print(Jacobi())
elif T == 2:
    print(GS())
elif T == 3:
    print(SOR())
else:
    print("Please Select 1,2 or 3")
