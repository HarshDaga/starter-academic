from Reviews import Reviews, Review
from Vocabulary import Vocabulary


def average_accuracy(models: list, data: list, smoothen: float = 1, min_occurrence: float = 0):
    accuracy = 0
    for i, model in enumerate(models):
        cur = model.accuracy(data[i], smoothen, min_occurrence)
        accuracy += cur
    accuracy /= len(models)
    return accuracy


class Model:
    def __init__(self, data: Reviews, vocab: Vocabulary):
        self.data: Reviews = data.copy()
        self.vocab = vocab

        self._compute_counts(self.data)

    def _compute_counts(self, reviews: Reviews):
        self.words = set()
        for review in reviews.all:
            self.words.update(review)

        self.positive_counts = {word: 0 for word in self.words}
        self.negative_counts = {word: 0 for word in self.words}

        for review in reviews.positive:
            for word in review:
                self.positive_counts[word] += 1
        for review in reviews.negative:
            for word in review:
                self.negative_counts[word] += 1

    def predict(self, review: Review, smoothen: float = 0, min_occurrence: float = 0):
        prob_pos, prob_neg = len(self.data.positive), len(self.data.negative)
        min_total = len(self.data.all) * min_occurrence
        for word in review:
            if self.vocab.is_stopword_index(word):
                continue

            positive = self.positive_counts.get(word, 0)  # self.data.count_positive(word)
            negative = self.negative_counts.get(word, 0)  # self.data.count_negative(word)
            total = positive + negative
            if total == 0 or total < min_total: # unknown or rare word
                continue

            prob_pos *= (positive + smoothen) / (total + smoothen * 2)  # |V| = 2
            prob_neg *= (negative + smoothen) / (total + smoothen * 2)
            if prob_pos == 0 or prob_neg == 0:
                break

        return 1 if prob_pos > prob_neg else -1

    def accuracy(self, reviews: Reviews, smoothen: float = 0, min_occurrence: float = 0):
        correct, total = 0, 0
        for review in reviews:
            if review.rating == self.predict(review, smoothen, min_occurrence):
                correct += 1
            total += 1

        return correct / total

    def top_words(self, top_count=10, min_occurrence=0.001):
        total_pos, total_neg = len(self.data.positive), len(self.data.negative)
        pos = total_pos / (total_pos + total_neg)
        neg = total_neg / (total_pos + total_neg)

        prob_pos, prob_neg, polarity = {}, {}, {}
        for word in self.words:
            if self.vocab.is_stopword_index(word):
                continue

            count_pos = self.positive_counts.get(word, 0)
            count_neg = self.negative_counts.get(word, 0)
            count_total = count_pos + count_neg
            if count_total < len(self.data.all) * min_occurrence:
                continue

            word_str = self.vocab.words[word]
            prob_pos[word_str] = pos * count_pos / count_total
            prob_neg[word_str] = neg * count_neg / count_total
            polarity[word_str] = (count_pos - count_neg ) / count_total

        top_pos = sorted(prob_pos.items(), key=lambda item: item[1], reverse=True)
        pos_words = [x[0] for x in top_pos]
        top_neg = sorted(prob_neg.items(), key=lambda item: item[1], reverse=True)
        neg_words = [x[0] for x in top_neg]

        return pos_words[:top_count], neg_words[:top_count]
