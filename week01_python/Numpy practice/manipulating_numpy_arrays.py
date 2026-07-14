import numpy as np

#initialise matrices
A = np.random.randint(0,10,(3,3))
B = np.random.randint(0,10,(3,3))
C = np.random.randint(0,10,(2,3))

#extracting rows and elements
row_1 = A[0,:]
print('First row: ' + str(row_1))
element_1 = A[0][0]
print('First element: ' + str(element_1))

#changing shape 
reshaped_C = C.reshape(3,2)
print('C reshaped: ' + str(reshaped_C))

#flattening
flattened_A = A.flatten()
print('A flattened: ' + str(flattened_A))
