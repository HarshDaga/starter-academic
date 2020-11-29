import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


class Vocabulary:
    def __init__(self, filename):
        with open(filename, encoding="utf8") as f:
            self.words = f.read().splitlines()
        self.reverse_map = {x: i for i, x in enumerate(self.words)}
        self.stopwords = set(stopwords.words('english'))

    def is_stopword(self, word):
        return word in self.stopwords

    def is_stopword_index(self, index):
        return self.words[index] in self.stopwords

    def get_index(self, item):
        return self.reverse_map.get(item, -1)
