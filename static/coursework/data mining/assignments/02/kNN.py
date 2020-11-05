from typing import Callable

import numpy as np
import pandas as pd
from Data import Data


class kNN:
    def __init__(self,
                 data: Data,
                 distance_func: Callable[[np.ndarray, np.ndarray], float]):
        self.data: Data = data.copy()
        self.distance_func = distance_func
        self.prediction_data = pd.concat([self.data.train, self.data.dev])

    def predict(self, point: np.ndarray, k):
        return self._predict_internal(self.prediction_data, point, k)

    def final_accuracy(self, k):
        points = self.data.test[self.data.features].to_numpy()
        success = 0
        for i, point in enumerate(points):
            pred = self.predict(point, k)
            real = self.data.test['class'].iloc[i]
            if pred == real:
                success += 1

        return success / points.shape[0]

    def k_accuracy(self, k):
        points = self.data.dev[self.data.features].to_numpy()
        success = 0
        for i, point in enumerate(points):
            pred = self._predict_internal(self.data.train, point, k)
            real = self.data.dev['class'].iloc[i]
            if pred == real:
                success += 1

        return success / points.shape[0]

    def _predict_internal(self, data, point: np.ndarray, k):
        distances = data[self.data.features].to_numpy()
        dist_transform = lambda x: self.distance_func(x, point)
        distances = np.apply_along_axis(dist_transform, 1, distances)

        neighbors = data.assign(d=distances).nsmallest(k, 'd')
        return neighbors['class'].mode()[0]
