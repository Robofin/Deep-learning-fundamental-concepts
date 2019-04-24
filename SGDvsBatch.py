import numpy as np 
import matplotlib.pyplot as plt

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

# Delta_Batch implements the delta rule
def Delta_Batch(W, X, D):
	alpha = 0.9
	start = 0
	stop = input_points

	for k in range(data_points):
		x = np.array([X[k]])
		d = np.array(D)[k]

		dWsum = np.transpose(np.zeros((3,1),np.float))

		""" 
			this for loop jumps 4 times starts at 0 to 3 to 6 to 9 since we 
			have 4 sets of inputs of 3 data points each.
			We get 12 weights in total, we get 3 at a time in the loop from 
			0, 1, 2 -> we get the 3 weights for these 3 data points, and the 
			3, 4, 5 follows the same pattern and so on till the 12th weight.
		"""
		j = 0
		for i in range(start, stop, input_points):
			j = i
			w = np.array([[W[0, i], W[0, i+1], W[0, i+2]]])
			v = np.dot(w, np.transpose(x))
			y = sigmoid(v)
			
			e = d-y
			
			delta = y * (1-y) * e
			
			# Delta rule
			dW = alpha * delta * x 

			dWsum = np.add(dWsum, dW)

		dWavg = dWsum/data_points
			
		W[0, j] = np.add(W[0, j], dWavg[0, 0])
		W[0, j+1] = np.add(W[0, j+1], dWavg[0, 1])
		W[0, j+2] = np.add(W[0, j+2], dWavg[0, 2])

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
W1 = 2*np.random.random((1, data_points*input_points)) - 1
W2 = W1

E1 = ''
E2 = ''

ErrDict1 =  {}
ErrDict2 =  {}

# 10000 epochs
for epoch in range(1000):
	W1 = Delta_SGD(W1, X, D)
	W2 = Delta_Batch(W2, X, D)
	es1 = 0;
	es2 = 0;

	start = 0
	stop = input_points

	# Test function
	for k in range(data_points):
		x = np.array([X[k]])

		for i in range(start, stop, input_points):
			w1 = np.array([[W1[0, i], W1[0, i+1], W1[0, i+2]]])
			v1 = np.dot(w1, np.transpose(x))
			y1 = sigmoid(v1)

			w2 = np.array([[W2[0, i], W2[0, i+1], W2[0, i+2]]])
			v2 = np.dot(w2, np.transpose(x))
			y2 = sigmoid(v2)

		if(stop <= data_points*input_points):
			start = stop
			stop += input_points

		es1 = es1 + (D[k] - y1[0][0]) ** 2
		es2 = es2 + (D[k] - y2[0][0]) ** 2

	ErrDict1[epoch] = es1 / data_points
	ErrDict2[epoch] = es2 / data_points

# line 1 points
lists1 = sorted(ErrDict1.items()) # sorted by key, return a list of tuples

X1, Y1 = zip(*lists1) # unpack a list of pairs into two tuples

# plotting the line 1 points 
plt.plot(X1, Y1, label = "SDG", color='blue')

# line 2 points
lists2 = sorted(ErrDict2.items()) # sorted by key, return a list of tuples

X2, Y2 = zip(*lists2) # unpack a list of pairs into two tuples

# plotting the line 2 points 
plt.plot(X2, Y2, '--', label = "BATCH", color='orange')
plt.xlabel('Epoch')

# Set the y axis label of the current axis.
plt.ylabel('Average training error')

# Set a title of the current axes.
plt.title('Comparison of the SGD and the Batch')

# Show a legend on the plot
plt.legend()

# Display a figure.
plt.show()

