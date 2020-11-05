import numpy as np
import pandas as pd

def k_fold(df: pd.DataFrame, k):
    n = df.shape[0] // k
    for i in range(0, df.shape[0], n):
        test = df.iloc[i:i + n]
        train = df.drop(test.index)
        yield train.copy(), test.copy()

class Data:
    features: list = ['sepal length', 'sepal width', 'petal length', 'petal width']
    columns: list = features + ['class']

    def __init__(self, df, train, dev, test):
        self._df: pd.DataFrame = df.copy()
        self.train: pd.DataFrame = train.copy()
        self.dev: pd.DataFrame = dev.copy()
        self.test: pd.DataFrame = test.copy()
        self.min = self.test[self.features].min()
        self.max = self.test[self.features].max()

    @classmethod
    def read(cls, filename: str, n_splits = 5, train_dev_split = 0.75):
        df = pd.read_csv(filename, header=None, names=cls.columns)
        shuffled = df.sample(frac=1)
        for train, test in k_fold(shuffled, n_splits):
            dev = train.sample(frac=1 - train_dev_split)
            train = train.drop(dev.index)

            yield cls(shuffled, train, dev, test)

    def normalized(self):
        result = self.copy()
        for df in (result._df, result.train, result.dev, result.test):
            df[result.features] = (df[result.features] - result.min) / (result.max - result.min)

        return result

    def normalize(self, points: np.ndarray):
        return (points - self.min.to_numpy()) / (self.max.to_numpy() - self.min.to_numpy())

    def copy(self):
        return Data(self._df, self.train, self.dev, self.test)
