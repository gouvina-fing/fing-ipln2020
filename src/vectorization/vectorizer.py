# DEPENDENCIES
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
import vectorization.features_vectorizer as features_vectorizer
import vectorization.embeddings_vectorizer  as embeddings_vectorizer
import util.const as const

# MAIN CLASS
class Vectorizer():
    '''
    Vectorizer interface
    '''

    # Constructor
    def __init__(self, vectorization=const.VECTORIZERS['word_embeddings']):

        # Type of vectorizer
        self.type = vectorization

        # Empty objects
        self.vectorizer = None
        self.dictionary = None

        # Depending on vectorization type, initializes different objects
        if self.type == const.VECTORIZERS['features']:
            self.vectorizer = DictVectorizer()
        if self.type == const.VECTORIZERS['word_embeddings']:
            self.dictionary = embeddings_vectorizer.get_dictionary()

    # Given a set X, vectorizes it and saves used vectorizer
    def fit(self, X):

        vectorized = []

        # If vectorization is by features, uses feature extraction and DictVectorizer
        if self.type == const.VECTORIZERS['features']:
            featurized = features_vectorizer.get_features(X)
            vectorized = self.vectorizer.fit_transform(featurized)

        # If vectorization is by word embeddings, uses word embeddings dictionary and mean
        if self.type == const.VECTORIZERS['word_embeddings']:
            vectorized = embeddings_vectorizer.get_vectors(X, self.dictionary)

        return vectorized

    # Given a set X, vectorizes it using last vectorizer
    def transform(self, X):
        
        vectorized = []      

        # If vectorization is by features, uses feature extraction and DictVectorizer        
        if self.type == const.VECTORIZERS['features']:
            featurized = features_vectorizer.get_features(X)
            vectorized = self.vectorizer.transform(featurized)

        # If vectorization is by word embeddings, uses word embeddings dictionary and mean        
        if self.type == const.VECTORIZERS['word_embeddings']:
            vectorized = embeddings_vectorizer.get_vectors(X, self.dictionary)

        return vectorized