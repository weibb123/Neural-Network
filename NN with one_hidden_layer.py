import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

# Define neural network layer_sizes
def layer_sizes(X, Y):
    """
    X -- input dataset of shape(input size, number of examples)
    Y -- labels of shape ( output size, number of examples)

    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    """
    n_x -- size of the input layer
    n_h -- size of hidden layer
    n_y -- size of output layer
    """
    return (n_x, n_h, n_y)

#Sigmoid Function
def sigmoid(x):

    s = 1/(1+np.exp(-x))

    return s

# Randomly Initialize model's parameters
def initialize_parameters(n_x, n_h, n_y):
    """
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# Forward propagation
def forward_propagation(X, parameters):

    """
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters

    return:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """

    # retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    #Implement forward propagation to calculate A2(output layer - probability)
    Z1 = W1.dot(X) + b1
    A1 = np.tanh(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2}

    return A2, cache

#Compute_cost using vectorization
def compute_cost(A2, Y):
    """
    Computes the cross-entropy cost given in equation
    
    A2 -- sigmoid output of the second activation
    Y -- "true" labels vector of shape

    return
    cost
    """
    m = Y.shape[1] # number of examples

    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1-A2))
    cost = -(1/m) * np.sum(logprobs)

    return cost

#Backpropagation to reduce loss
def backward_propagation(parameters, cache, X, Y):
    """
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape
    Y -- "true" labels vector of shape

    return
    grads -- python dictionary containing gradients
    """
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    #Retrieve A1 and A2 from dictionary "cache"
    A1 = cache["A1"]
    A2 = cache["A2"]

    #back propagation with equations
    dZ2 = A2 - Y
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims = True)
    dZ1 = W2.T.dot(dZ2) * (1 - np.power(A1, 2))
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1, axis = 1, keepdims = True)

    grads = {"dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2}
    
    return grads

#change update_parameters using propagation
def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    return
    parameters -- python dictionary containing updated parameters
    """
    # retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    #update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2}

    return parameters

# build neural network model
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):

    """
    X -- dataset of shape 
    Y -- labels of shape
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print cost every 1000 iterations

    return:
    parameters -- parameters learn by the model.
    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        #forward propagation
        A2, cache = forward_propagation(X, parameters)
        # cost 
        cost = compute_cost(A2, Y)
        # backpropagation
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent (update parameter)
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print ("cost after iteration %i: %f" %(i, cost))
    
    return parameters

# Predict with my model
def predict(parameters, X):

    """
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    A2, cache = forward_propagation(X, parameters)
    prediction = A2 > 0.5 # A2 is our output layer !, because we only have one hidden layer

    return prediction

# test model on (XXXXX) dataset

# parameters = nn_model(X, Y, n_h = 4, num_iterations = 1000, print_cost = True)
#plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
#plt.title("Decision boundary for hidden layer size")