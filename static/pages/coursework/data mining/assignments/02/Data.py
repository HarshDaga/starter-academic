import pandas as pd


class Data:
    def __init__(self, filename, split_frac):
        self.features = ['sepal length', 'sepal width', 'petal length', 'petal width']
        self.columns = self.features + ['class']
        self._df = pd.read_csv(filename, header=None, names=self.columns)
        self.dev = self._df.sample(frac=split_frac)
        self.test = self._df.drop(self.dev.index)
