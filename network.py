import numpy as np

class som_network():
    def __init__(self, input_size, som_m, som_n):
        self.input = None
        self._m = som_m
        self._n = som_n
        self.weights = np.random.uniform(-1, 1, (self._m, self._n, input_size))
        self.input_dim = input_size
        """
        self.p = np.random.rand(self._m, self._n)
        self.B = 0.2
        self.C = 0.2
        """
        self.position = np.zeros((self._m, self._n, 2))
        for i in range(self._m):
            for j in range(self._n):
                self.position[i][j][0] = i
                self.position[i][j][1] = j

    def forward(self, input):
        # select the winner
        self.input = input
        #self.b = self.C * (1 / (self._m * self._n * self.input.shape[1]) - self.p)
        #Min = np.linalg.norm(self.input - self.weights[0][0]) - self.b[0][0]

        distances = np.linalg.norm(self.weights - self.input.T, axis=2)
        idx_i, idx_j = np.unravel_index(distances.argmin(), distances.shape)
        """
        self.p[idx_i][idx_j] += self.B * (1 - self.p[idx_i][idx_j])
        for i in range(self._m):
            for j in range(self._n):
                self.p[i][j] += self.B * (0 - self.p[i][j])
        """
        return idx_i, idx_j

    def backward(self, win_i, win_j, neighbor_raduis, learning_rate):
        # update winner & neighborhood
        """
        for i in range(self._m):
            for j in range(self._n):
                distance = np.linalg.norm(np.array([i, j]) - np.array([win_i, win_j]))
                output_gradient = np.exp(- distance**2 / (2 * neighbor_raduis**2))
                # squeeze: (2, 1) -> (2, )
                self.weights[i][j] += learning_rate * output_gradient * (np.squeeze(self.input) - self.weights[i][j])
        """
        distances = np.linalg.norm(self.position - np.array([win_i, win_j]), axis=2)
        out = np.exp(- distances**2 / (2 * neighbor_raduis**2))
        tmp = np.reshape(np.squeeze(self.input), (1, 1, self.input_dim))
        self.weights += learning_rate * (out[:, :, None] * (tmp - self.weights))
