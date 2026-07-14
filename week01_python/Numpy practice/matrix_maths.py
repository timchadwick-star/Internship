import numpy as np

#initialise matrices
A = np.random.randint(0,10,(3,3))
B = np.random.randint(0,10,(3,3))

#initialise vectors
v = np.random.randint(0,10,3)
u = np.random.randint(0,10,3)

#dot product
w = np.dot(u,v)
print('Dot product of u and v is: ' + str(w))

#matrix multiplication
D = np.matmul(A,B)
print('A x B = ' + str(D))

#determinant
det_A = np.linalg.det(A)
print('Determinant of A = ' + str(det_A))

#matrix applied to vector
q = np.matmul(A,v)
print('A x v = ' + str(q))

