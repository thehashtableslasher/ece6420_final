from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.base import BaseEstimator, TransformerMixin

def vectorize(doc):
    pass # TODO

class D2VVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.model = None
        self.hyperparameters = kwargs
    
    def fit(self, X, y=None):
        self.model = model = Doc2Vec([TaggedDocument(tokenize(x), [i]) for i, x in enumerate(X)], **self.hyperparameters)
        return self

    def transform(self, X, y=None):
        return np.concat([self.model.infer_vector(tokenize(x)) for x in X])

