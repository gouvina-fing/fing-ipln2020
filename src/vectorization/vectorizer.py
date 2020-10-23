# DEPENDENCIAS (Bibliotecas)
# ---------------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer

# DEPENDENCIAS (Locales)
# ----------------------------------------------------------------------------------------------------
import vectorization.features_vectorizer as features_vectorizer
import vectorization.embeddings_vectorizer  as embeddings_vectorizer
import util.const as const

# CLASE PRINCIPAL
# ----------------------------------------------------------------------------------------------------
class Vectorizer():

    # Constructor
    def __init__(self, vectorization=const.VECTORIZERS['embeddings']):

        # Type of vectorizer
        self.type = vectorization

        # Interfaces auxiliares
        self.vectorizer = None
        self.dictionary = None

        # Dependiendo del tipo de vectorización, inicializa distintas interfaces
        if self.type == const.VECTORIZERS['features']:
            self.vectorizer = DictVectorizer()
        if self.type == const.VECTORIZERS['embeddings']:
            self.dictionary = embeddings_vectorizer.get_dictionary()

    # Dado un conjunto de ejemplos X, adapta el vectorizador al mismo y vectoriza X
    def fit(self, X):

        vectorized = []

        # Extrae features y los vectoriza con DictVectorizer
        if self.type == const.VECTORIZERS['features']:
            featurized = features_vectorizer.get_features(X)
            vectorized = self.vectorizer.fit_transform(featurized)

        # Traduce a embeddings y calcula vector promedio
        if self.type == const.VECTORIZERS['embeddings']:
            vectorized = embeddings_vectorizer.get_vectors(X, self.dictionary)

        return vectorized

    # Dado un conjunto de ejemplos X, utilizando el vectorizador preadaptado, vectoriza X
    def transform(self, X):
        
        vectorized = []      

        # Extrae features y los vectoriza con DictVectorizer     
        if self.type == const.VECTORIZERS['features']:
            featurized = features_vectorizer.get_features(X)
            vectorized = self.vectorizer.transform(featurized)

        # Traduce a embeddings y calcula vector promedio     
        if self.type == const.VECTORIZERS['embeddings']:
            vectorized = embeddings_vectorizer.get_vectors(X, self.dictionary)

        return vectorized