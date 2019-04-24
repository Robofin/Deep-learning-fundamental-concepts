import numpy as np 

# Number of all data points i.e. Total No. of our sample dataset items
data_points = 4 
# Number of inputs points only (without correct-output)
input_points = 3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

# Delta_SGD implements the delta rule
def Delta_SGD(W, X, D):
    alpha = 0.9
    start = 0
    stop = input_points

    for k in range(data_points):
        x = np.array([X[k]])
        d = np.array(D)[k]

        """ 
            this for loop jumps 4 times starts at 0 to 3 to 6 to 9 since we 
            have 4 sets of inputs of 3 data points each.
            We get 12 weights in total, we get 3 at a time in the loop from 
            0, 1, 2 -> we get the 3 weights for these 3 data points, and the 
            3, 4, 5 follows the same pattern and so on till the 12th weight.
        """
        for i in range(start, stop, input_points):
            w = np.array([[W[0, i], W[0, i+1], W[0, i+2]]])
            v = np.dot(w, np.transpose(x))
            y = sigmoid(v)
            
            e = d-y
            
            delta = y * (1-y) * e
            
            # Delta rule
            dW = alpha * delta * x 
            
            W[0, i] = np.add(W[0, i], dW[0, 0])
            W[0, i+1] = np.add(W[0, i+1], dW[0, 1])
            W[0, i+2] = np.add(W[0, i+2], dW[0, 2])

        if(stop <= data_points*input_points):
            start = stop
            stop += input_points

    return W
    
# Number of Input data points = N
# Each data point consists of an input and correct-output pair
# In this code we have 3 inputs, 1 correct-output

# Input data points
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])

# Correct output pairs
D = np.transpose(np.array([[0,0,1,1]]))

# learning rate
alpha = 0.9

# Initialize weights with weights real numbers between -1 and 1
W =  2*np.random.random((1, data_points*input_points)) - 1

# 10000 epochs
for epoch in range(10000):
    W = Delta_SGD(W, X, D)

start = 0
stop = input_points

# Test function
for k in range(data_points):
    x = np.array([X[k]])

    for i in range(start, stop, input_points):
        w = np.array([[W[0, i], W[0, i+1], W[0, i+2]]])
        v = np.dot(w, np.transpose(x))
        y = sigmoid(v)

    if(stop <= data_points*input_points):
        start = stop
        stop += input_points

    print("{}Neuron: {}, Output: {}, Correct Output: {} {}".format("\n", k+1, y, D[k], "\n"))
