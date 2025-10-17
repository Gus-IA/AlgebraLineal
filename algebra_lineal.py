# valor escalar

x = 1
print(x)

# vector
import numpy as np

y = np.array([1, 2, 3, 4])
print(y)
print(y[0])


# matriz
l = [[1, 2], [3, 4]]
A = np.array(l)
print(A)
print(A[0])

# tensores
B = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
print(B)

# matriz traspuesta
C = np.arange(10).reshape(2,5)
print(C)

print(C.T)


# suma matriz

D = np.random.rand(3,3)
E = np.random.rand(3,3)

print(D, E)

print(D + E)


# multiplicación matriz

F = np.array([[1,2,1],[0,1,0],[2,3,4]])
G = np.array([[2,5],[6,7],[1,8]])

print(F, G)

# producto de matriz
I = F.dot(G)
print(I)

I = F @ G
print(I)


# multiplicamos cada elemento de manera independiente
J = np.random.rand(2,2)
K = np.random.rand(2,2)
print(J*K)


# matriz identidad

M = np.eye(3)
print(M)


# matriz inversa
import numpy.linalg as linalg

N = np.array([[1,2,3],[5,6,7],[21,29,31]])
print(N)

linalg.inv(N)


P = np.array([[2,6], [5,3]])
Q = np.array([6, -9])

R = linalg.inv(P).dot(Q)
print(R)


# descomposición matriz

T = np.array([[1,2,3],[5,7,11],[21,29,31]])
print(T)

U, V = linalg.eig(T)
print(U, V)