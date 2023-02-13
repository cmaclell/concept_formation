import numpy as np

class MultiNB:

    def __init__(self):
        self.alpha = 1 

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n = X.shape[0]

        X_by_class = [X[y == c] for c in np.unique(y)]
        self.prior = np.array([len(X_class) / n for X_class in X_by_class])

        self.word_counts = np.array([sub_arr.sum(axis=0) for sub_arr in X_by_class]) + self.alpha
        self.lk_word = self.word_counts / self.word_counts.sum(axis=1).reshape(-1, 1)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """ Predict probability of class membership """

        # loop over each observation to calculate conditional probabilities
        class_numerators = np.zeros(shape=(X.shape[0], self.prior.shape[0]))
        for i, x in enumerate(X):
            word_exists = x.astype(bool)
            lk_words_present = self.lk_word[:, word_exists] ** x[word_exists]
            # print("shape", self.word_counts.shape)
            # print(self.word_counts[:, word_exists])
            # print(self.word_counts.sum(axis=1).reshape(-1, 1))
            lk_message = np.log(lk_words_present).sum(axis=1)
            class_numerators[i] = lk_message + np.log(self.prior)

        # normalize_term = class_numerators.sum(axis=1).reshape(-1, 1)
        conditional_probas = class_numerators
        # conditional_probas = class_numerators / normalize_term
        # assert (conditional_probas.sum(axis=1) - 1 < 0.001).all(), 'Rows should sum to 1'
        return conditional_probas



