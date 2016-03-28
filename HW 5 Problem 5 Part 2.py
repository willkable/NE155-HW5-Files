import numpy as np
import matplotlib.pyplot as plt


# What I am doing is creating a plot of the number of iterations needed of the SOR method vs. the Omega value for the
# method. I am doing this over a meshed space of Omega values. I am looking to see if there are
# any trends between the number of iterations needed and the omega value. I also am using the same n=5, tolerance
# 10^-6 and initial 0 guess matrix we used before. So what we can do is try to find the best Omega value that will
# solve our system with this tolerance in the shortest number of iterations


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


def SOR(w):
    A, B, x = Matrix()
    Upper = np.triu(A, 1)
    Lower = np.tril(A, -1)
    Middle = np.diagflat(np.diag(A))
    DiagInverse = np.linalg.inv(Middle + Lower * w)
    iters = 0
    Max_Iters = 100000
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
        if error < 10**-6:
            return iters

wspace = np.linspace(0.1, 1.95, 100)
yspace = []
Min = 1000000
Loc = 0
for i in range(len(wspace)):
    yspace.append(SOR(wspace[i]))
    if SOR(wspace[i]) < Min:
        Min = SOR(wspace[i])
        Loc = i

print("Minimum Number of Iterations Required: ", Min)
print("Omega Value at Minimum: ", wspace[Loc])


fig = plt.figure(figsize=(16, 9))
plt.plot(wspace, yspace, 'bo', label="Error Values")
plt.xlabel('Omega Values', size=18)
plt.ylabel('Number of Iterations Required', size=18)
plt.title('Search for the Optimum Omega Value', size=22)
plt.show()
