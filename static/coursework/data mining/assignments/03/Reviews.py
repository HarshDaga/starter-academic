import re
from random import shuffle


class Review:
    _regex = re.compile(r"((\d+):(\d+))")

    def __init__(self, rating: int, freq: dict):
        self.rating_actual: int = rating
        self.rating: int = 1 if rating > 5 else -1
        self.freq: dict = freq.copy()

    def __contains__(self, item):
        return item in self.freq

    def __iter__(self):
        return iter(self.freq.keys())

    @classmethod
    def parse(cls, line: str):
        line = line.split(None, 1)
        rating = int(line[0])
        words = '' if len(line) == 1 else line[1]
        matches = list(cls._regex.finditer(words))

        freq = {}
        for match in matches:
            ind, times = match.group(2), match.group(3)
            freq[int(ind)] = int(times)

        return cls(rating, freq)


class Reviews:
    def __init__(self, reviews: list):
        self.all = reviews.copy()
        self.positive = [r for r in self.all if r.rating_actual > 5]
        self.negative = [r for r in self.all if r.rating_actual < 5]

    def __iter__(self):
        return iter(self.all)

    @classmethod
    def parse(cls, filename: str):
        reviews = []
        with open(filename) as f:
            for _, line in enumerate(f):
                review = Review.parse(line)
                reviews.append(review)

        return cls(reviews)

    def shuffle(self):
        copy = self.all.copy()
        shuffle(copy)
        return Reviews(copy)

    def count(self, index):
        return sum(map(lambda x: index in x, self.all))

    def count_positive(self, index):
        return sum(map(lambda x: index in x, self.positive))

    def count_negative(self, index):
        return sum(map(lambda x: index in x, self.negative))

    def copy(self):
        return Reviews(self.all)