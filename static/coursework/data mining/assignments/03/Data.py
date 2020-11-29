from random import shuffle

from Reviews import Reviews


def k_fold(l: list, k):
    n = len(l) // k
    for i in range(0, len(l), n):
        test = l[i:i + n].copy()
        train = l.copy()
        del train[i:i + n]
        yield train, test


class Data:
    def __init__(self, train: Reviews, dev: Reviews, test: Reviews):
        self.train: Reviews = train.copy()
        self.dev: Reviews = dev.copy()
        self.test: Reviews = test.copy()
        self.all_train: Reviews = Reviews(self.train.all + self.dev.all)

    @classmethod
    def read_train(cls, folder: str, n_splits=5):
        train = Reviews.parse(rf'{folder}\\train\\labeledBow.feat').shuffle()

        for train2, dev in k_fold(train.all, n_splits):
            yield cls(Reviews(train2), Reviews(dev), Reviews([]))

    @classmethod
    def read_all(cls, folder: str, n_splits=5):
        train = Reviews.parse(rf'{folder}\\train\\labeledBow.feat')
        test = Reviews.parse(rf'{folder}\\test\\labeledBow.feat')
        all_reviews = train.all + test.all
        shuffle(all_reviews)

        for train, test in k_fold(all_reviews, n_splits):
            yield cls(Reviews(train), Reviews([]), Reviews(test))

    def copy(self):
        return Data(self.train, self.dev, self.test)
