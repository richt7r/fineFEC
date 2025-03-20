import numpy as np
from numba import jit

@jit(nogil=True)
def spa(data,i1,i2,i3,i4,num_iterations):
    message_vector = np.zeros(i2.size,dtype=np.float32)
    for i,pos in enumerate(i2):
        message_vector[i] = data[pos]
    iteration = 0
    while iteration < num_iterations:
        for i in range (i1.size-1):
            prod = np.tanh(message_vector[i1[i]:i1[i+1]]/2).prod()
            message_vector[i1[i]:i1[i+1]] = np.log((1+(prod / np.tanh(message_vector[i1[i]:i1[i+1]]/2)))/(1-(prod / np.tanh(message_vector[i1[i]:i1[i+1]]/2))))
        iteration+=1
        if iteration == num_iterations:
            for i in range (i3.size-1):
                data[i2[i4[i3[i]]]]+= message_vector[i4[i3[i]:i3[i+1]]].sum()
            return data
        for i in range (i3.size-1):
            sum = message_vector[i4[i3[i]:i3[i+1]]].sum() + data[i2[i4[i3[i]]]]
            message_vector[i4[i3[i]:i3[i+1]]] = sum - message_vector[i4[i3[i]:i3[i+1]]]

