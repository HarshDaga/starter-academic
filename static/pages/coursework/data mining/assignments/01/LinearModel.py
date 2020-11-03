import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from Data import Data


class LinearModel:
    def __init__(self, degree: int, hyper_parameter: float = 0):
        self.degree = degree
        self.hyper_parameter = hyper_parameter
        self.model = None
        self.weights = np.array([])

    def train(self, data: Data):
        self.model = make_pipeline(PolynomialFeatures(self.degree),
                                   Ridge(alpha=self.hyper_parameter, solver='auto'))
        self.model.fit(data.X_train[:, np.newaxis], data.Y_train)
        self.weights = self.model.steps[-1][1].coef_.copy()
        self.weights[0] = self.model.steps[-1][1].intercept_
        return self.model

    def predict(self, X):
        return self.model.predict(X[:, np.newaxis])

    def error(self, X, y):
        y_pred = self.predict(X)
        return np.sqrt(mean_squared_error(y, y_pred))

    def plot(self, ax, data: Data):
        x_plot = np.linspace(0, 1, 100)
        y_plot = self.predict(x_plot)

        ax.plot(data.X, data.ground_truth, color='lime')
        ax.scatter(data.X_train, data.Y_train, facecolors='none', edgecolors='b')
        ax.plot(x_plot, y_plot, color='red', label=f'M = {self.degree}')
        ax.set_xlim([None, None])
        ax.set_ylim([-1.5, 1.5])
        ax.legend(loc='upper right')

        return ax