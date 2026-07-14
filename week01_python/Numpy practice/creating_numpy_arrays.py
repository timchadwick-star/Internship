import numpy as np

#create arrays from python lists
python_list = [1,2,3,4]
python_2d_list = [[1,2,3,4],[5,6,7,8]]

#create numpy arrays from python lists
numpy_array = np.array(python_list)
numpy_2d_array = np.array(python_2d_list)

# print(numpy_array)
# print(numpy_2d_array)


#initialise arrays directly
array_float = np.zeros((2,3),dtype=float)
array_int = np.zeros((2,3),dtype=int)

# print(array_float)
# print(array_int)

#convert numpy arrays back to python lists
list_from_array = numpy_2d_array.tolist()
#print(list_from_array)

#finding shape of arrays

#print(numpy_2d_array.shape)


