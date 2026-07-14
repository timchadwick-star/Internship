import numpy as np

#initialise arrays
array_1 = np.random.randint(0,10,(2,3))
array_2 = np.random.randint(0,10,(2,3))

#addition
array_sum = array_1 + array_2
print('Array sum = ' + str(array_sum))

#subtraction
array_diff = array_1 - array_2
print('Array difference = ' + str(array_diff))

#multiplication my scalar
multiplied_array = 2 * array_1

#division by scalar
divided_array = array_1 / 2

#exponentiation
array_power_2 = array_1**2
