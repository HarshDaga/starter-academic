from typing import Callable, List

import numpy as np
import pandas as pd


class kNN:
    def __init__(self,
                 dev_data: pd.DataFrame,
                 features: List[str],
                 distance_func: Callable[[np.ndarray, np.ndarray], float],
                 use_normalization=False):
        self.dev = dev_data.copy()
        self.features = features
        self.distance_func = distance_func
        self.use_normalization = use_normalization
        self.min, self.max = 0, 1  # (0, 1) defaults behave the same as no normalization

        if self.use_normalization:
            df = self.dev[self.features]
            self.min = df.min().to_numpy()
            self.max = df.max().to_numpy()
            self.dev[self.features] = (df - df.min()) / (df.max() - df.min())

    def normalize(self, point: np.ndarray):
        return (point - self.min) / (self.max - self.min)

    def predict(self, point: np.ndarray, k):
        return self._predict_internal(self.dev, self.normalize(point), k)

    def calc_accuracy(self, test_data: pd.DataFrame, k):
        points = test_data[self.features].to_numpy()
        success = 0
        for i, point in enumerate(points):
            pred = self.predict(point, k)
            real = test_data['class'].iloc[i]
            if pred == real:
                success += 1

        return success / points.shape[0]

    def k_accuracy(self, k):
        points = self.dev[self.features].to_numpy()
        success = 0
        for i, point in enumerate(points):
            pred = self._predict_internal(self.dev.drop(self.dev.index[i]), point, k)
            real = self.dev['class'].iloc[i]
            if pred == real:
                success += 1

        return success / points.shape[0]

    def _predict_internal(self, data, point: np.ndarray, k):
        distances = data[self.features].to_numpy(copy=True)
        dist_transform = lambda x: self.distance_func(x, point)
        distances = np.apply_along_axis(dist_transform, 1, distances)

        neighbors = data.assign(d=distances).sort_values('d')[:k]
        return neighbors['class'].mode()[0]
