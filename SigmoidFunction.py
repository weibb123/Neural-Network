import numpy as np

def sigmoid(x):

    s = 1/(1+np.exp(-x))

    return s

# test cases
t_x = np.array([1, 2, 3])
print("sigmoid(t_x) = " + str(sigmoid(t_x)))

#function for computing derivative of sigmoid
def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s*(1-s)

    return ds

#test cases
print ("sigmoid_derivative(t_x) = " + str(sigmoid_derivative(t_x)))
