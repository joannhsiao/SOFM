import numpy as np
from network import som_network
import math
from random import shuffle
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt

def predict(network, data):
    output = data
    win_i, win_j = network.forward(output)
    return win_i, win_j

def train(network, x_train, neighbor_radius, epochs=1000, learning_rate=0.5):
	nr0 = neighbor_radius
	lr0 = learning_rate
	for epoch in range(epochs):
		neighbor_radius = nr0 * math.exp(-(epoch / epochs))
		learning_rate = lr0 * math.exp(-(epoch / epochs))
		for x in x_train:
			# get the winner
			win_i, win_j = predict(network, x)
			# update the winner & neighborhood
			network.backward(win_i, win_j, neighbor_radius, learning_rate)
		if epoch % 100 == 0:
			print("epoch {}, step: {}, radius: {}".format(epoch, learning_rate, neighbor_radius))
	print("SOM training completed")

def data_process(file):
	with open(file, "r") as f:
		lines = f.readlines()
	tmp = []
	for num, line in enumerate(lines):
		tmp.append(line)
	shuffle(tmp)
	data = []
	y = []
	for line in tmp:
		xdata = line.split()
		y.append(int(float(xdata[-1])))
		for i in range(len(xdata)):
			xdata[i] = float(xdata[i])
		data.append(xdata[0:-1])
		input_dim = len(xdata) - 1
	del tmp

	count_y = {}
	for i in range(len(y)):
		if y[i] not in count_y:
			count_y[y[i]] = 1
		else:
			continue
	classes = len(count_y)
	del count_y

	# normalization
	x_train = (data - np.min(data)) / (np.max(data) - np.min(data))

	x_train = np.reshape(data, (len(data), input_dim, 1))
	y_train = np.reshape(y, (len(y), 1))
	return x_train, y_train, input_dim, classes

def process(file, epoch):
	x_train, y_train, input_dim, classes = data_process(file)
	num_m, num_n = 10, 10
	network = som_network(input_dim, num_m, num_n)

	lr0 = 0.9
	nr0 = 10
	train(network, x_train, nr0, epoch, lr0)

	# store weights
	weights = network.weights.ravel().tolist()
	w_tmp = []
	for i in range(0, len(weights), input_dim):
		w_tmp.append(weights[i:i+input_dim])
	weights = np.array(w_tmp)
	weights = np.reshape(weights, (num_m, num_n, input_dim))
	
	print("topology processing...")
	Map = np.zeros((num_m, num_n))
	for i in range(num_m):
		for j in range(num_n):
			Min = -1
			for x, y in zip(x_train, y_train):
				if np.linalg.norm(np.squeeze(x) - weights[i][j]) < Min or  Min == -1:
					Min = np.linalg.norm(np.squeeze(x) - weights[i][j])
					Map[i][j] = y
	
	print(Map)
	
	weights = np.reshape(weights, (num_m*num_n, input_dim))
	if weights.shape[1] > 2:
		pca = PCA(n_components=2, iterated_power=1)
		weights = pca.fit_transform(weights)
	
	Map = np.reshape(Map, (num_m*num_n, 1))
	return weights, Map

"""
if __name__ == "__main__":
	weights, Map = process("Dataset/2CS.txt", 1000)
	
	plt.scatter(weights[:, 0], weights[:, 1], c=Map, s=10, cmap='coolwarm')
	plt.show()
"""