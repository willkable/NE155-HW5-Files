import numpy as np
import matplotlib.pyplot as plt


def tridiag(z, bot, mid, top):
    a = [bot] * (z - abs(-1))
    b = [mid] * (z - abs(0))
    c = [top] * (z - abs(1))
    return np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)


def Bmatrix(Size, B=[]):
    for i in range(0, Size):
        B.append(i)
    B = np.transpose(B)
    B = B.reshape(Size, 1)
    return B

print("A Matrix: ")
A = tridiag(100, -1, 2, -1)
print(A)
print("B Matrix: ")
B = Bmatrix(100)
print(B)
Cond = np.linalg.cond(A)
print("Condition Number: ", Cond)
inv = np.linalg.inv(A)
sol = inv * B
print("Matrix Solution with Inverse: ")
Sol1 = np.dot(inv, B)
print(Sol1)
print("Matrix Solution with Numpy: ")
Sol2 = np.linalg.solve(A, B)
print(Sol2)

fig = plt.figure(figsize=(16, 9))
xspace = np.linspace(0, 100, 100)
plt.plot(xspace, Sol1, "r-", label="Explict Solution")
plt.plot(xspace, Sol2, "bo", label="Numerical Solution")
plt.legend()
plt.xlabel('X Number [1-100]', size=18)
plt.ylabel('Solution', size=18)
plt.title('Comparison of Explicit to Numerical Solution for Matrix', size=22)
plt.show()
