# DEPENDENCIAS (Bibliotecas)
# ---------------------------------------------------------------
import sys
import time
import pandas as pd
import numpy as np
import statistics
import re
from tokenizer import tokenize

# DEPENDENCIAS (Locales)
# ----------------------------------------------------------------------------------------------------
import modules.const as const

# CLASE PRINCIPAL
# ----------------------------------------------------------------------------------------------------
class Vectorizer():

    # FUNCIONES AUXILIARES
    # ----------------------------------------------------------------------------------------------------

    # Devuelve diccionario de embeddings
    def get_dictionary(self):
        dicc = {}

        # Lectura de archivo de embeddings como pandas DataFrame
        tic = time.time()
        df_test = pd.read_csv(const.DATA_FOLDER + const.EMBEDDINGS_FILE, engine='python', sep='\s+', header=None)
        toc = time.time()
        print('(VECTORIZER) Embeddings dictionary loaded in ' + '{0:.2f}'.format(toc - tic) + ' seconds')

        # Obtener palabras y valores de embeddings
        rows = df_test.shape[0] - 1
        words = df_test.loc[0:rows, 0]
        df = df_test.loc[0:rows, 1:300]

        # Reemplazar NaN e inf con 0
        df[df==np.inf] = np.nan
        df.fillna(0)

        # Llenar diccionario con cada palabra
        tic = time.time()
        for index, row in df.iterrows():
            if not(words[index] in dicc.keys()):
                dicc[words[index]] = row.values
        toc = time.time()
        print('(VECTORIZER) Embeddings dictionary adapted in ' + '{0:.2f}'.format(toc - tic) + ' seconds')

        return dicc

    # Constructor
    def __init__(self):

        # Interfaces auxiliares
        self.dictionary = self.get_dictionary()

    # FUNCIONES PRINCIPALES
    # ----------------------------------------------------------------------------------------------------

    # Dado un conjunto de ejemplos X, utilizando el vectorizador preadaptado, vectoriza X
    def transform(self, X):
        X['texto'] = X['texto'].apply(self.convert_tweet_to_embedding, embeddings=self.dictionary)
        return X
    
    # FUNCIONES AUXILIARES
    # ----------------------------------------------------------------------------------------------------

    # Dado un tweet y un diccionario, devuelve el tweet como un vector media de los vectores de cada palabra
    def convert_tweet_to_embedding(self, tweet, embeddings):
        words = np.array(self.tokenize_text(tweet), dtype=object)
        return self.mean_of_tweet_embedding(words, embeddings)

    # Dado un texto, lo devuelve como una lista de palabras tokenizadas
    def tokenize_text(self, text):

        # Erase non alphanumerical characters
        regex = r"¡|!|,|\?|\.|=|\+|-|_|&|\^|%|$|#|@|\(|\)|`|'|<|>|/|:|;|\*|$|¿|\[|\]|\{|\}|~"
        text = re.sub(regex, ' ', text)

        # Tokenize
        words = [ token.txt for token in tokenize(text) if token.txt is not None]
        return words

    # Dada una lista de palabras y un diccionario, devuelve un vector media representando la lista de palabras
    def mean_of_tweet_embedding(self, words, embeddings):
        data = pd.Series(words)
        data = data.apply(self.token_to_embedding, embeddings=embeddings)
        each_index_array = list(zip(*data))
        each_index_array[0] = each_index_array[1]
        each_index_array = list(map(statistics.mean, each_index_array))
        each_index_array = np.array(each_index_array)

        return each_index_array

    # Dada una palabra y un diccionario, devuelve un vector representando la palabra 
    def token_to_embedding(self, word, embeddings):
        if word in embeddings.keys():
            return embeddings.get(word)
        else:
            return np.zeros(300)

