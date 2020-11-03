import numpy as np


class Data:
    def __init__(self, size):
        self.size = size
        self.X = np.linspace(0, 1, size)
        self.noise = np.random.normal(0, 0.1, size)
        self.ground_truth = np.sin(self.X * 2 * np.pi)
        self.Y = self.ground_truth + self.noise

        self.X_test = self.X[0::2]
        self.X_train = self.X[1::2]
        self.Y_test = self.Y[0::2]
        self.Y_train = self.Y[1::2]