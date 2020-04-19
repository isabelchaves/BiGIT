from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfImplementation:
    def __init__(self, corpus):
        self.model = self.train(corpus=corpus)

    def train(self, corpus):
        corpus = list(dict.fromkeys(corpus))
        model = TfidfVectorizer(stop_words='english').fit(corpus)
        return model
