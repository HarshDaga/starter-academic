import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


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
        kf = KFold(n_splits=n_splits, shuffle=True)
        for train_index, test_index in kf.split(df):
            train = df.loc[train_index]
            test = df.loc[test_index]
            train, dev = train_test_split(train, train_size=train_dev_split)

            yield cls(df, train, dev, test)

    def normalized(self):
        result = self.copy()
        for df in (result._df, result.train, result.dev, result.test):
            df[result.features] = (df[result.features] - result.min) / (result.max - result.min)

        return result

    def normalize(self, points: np.ndarray):
        return (points - self.min.to_numpy()) / (self.max.to_numpy() - self.min.to_numpy())

    def copy(self):
        return Data(self._df, self.train, self.dev, self.test)
