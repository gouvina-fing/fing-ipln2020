# DEPENDENCIAS (Bibliotecas)
# ---------------------------------------------------------------
import sys
import pandas as pd
import numpy as np
import statistics
import re
from tokenizer import tokenize

# DEPENDENCIAS (Locales)
# ----------------------------------------------------------------------------------------------------
import util.const as const

# FUNCIONES PRINCIPALES
# ----------------------------------------------------------------------------------------------------

# Dada una lista de tweets y un diccionario, devuelve lista de tweets vectorizados
def get_vectors(tweets, dictionary):
    # aux = tweets['texto'].apply(convert_tweet_to_embedding, embeddings=dictionary)

    tweets['texto'] = tweets['texto'].apply(convert_tweet_to_embedding, embeddings=dictionary)
    return tweets

# Devuelve diccionario de embeddings
def get_dictionary():
    dicc = {}

    # Lectura de archivo de embeddings como pandas DataFrame
    df_test = pd.read_csv(const.DATA_FOLDER + const.EMBEDDINGS_FILE, engine='python', sep='\s+', header=None) 

    # Obtener palabras y valores de embeddings
    rows = df_test.shape[0] - 1
    words = df_test.loc[0:rows, 0]
    df = df_test.loc[0:rows, 1:300]

    # Reemplazar NaN e inf con 0
    df[df==np.inf] = np.nan
    df.fillna(0)

    # Llenar diccionario con cada palabra
    for index, row in df.iterrows():
        if not(words[index] in dicc.keys()):
            dicc[words[index]] = row.values
    
    return dicc

# FUNCIONES AUXILIARES
# ----------------------------------------------------------------------------------------------------

# Dado un tweet y un diccionario, devuelve el tweet como un vector media de los vectores de cada palabra
def convert_tweet_to_embedding(tweet, embeddings):
    words = np.array(tokenize_text(tweet), dtype=object)
    return mean_of_tweet_embedding(words, embeddings)

# Dado un texto, lo devuelve como una lista de palabras tokenizadas
def tokenize_text(text):

    # Erase non alphanumerical characters
    regex = r"¡|!|,|\?|\.|=|\+|-|_|&|\^|%|$|#|@|\(|\)|`|'|<|>|/|:|;|\*|$|¿|\[|\]|\{|\}|~"
    text = re.sub(regex, ' ', text)

    # Tokenize
    words = [ token.txt for token in tokenize(text) if token.txt is not None]
    return words

# Dada una lista de palabras y un diccionario, devuelve un vector media representando la lista de palabras
def mean_of_tweet_embedding(words, embeddings):
    data = pd.Series(words)
    data = data.apply(token_to_embedding, embeddings=embeddings)
    each_index_array = list(zip(*data))
    each_index_array[0] = np.zeros(1)
    each_index_array = list(map(statistics.mean, each_index_array))
    each_index_array = np.array(each_index_array)

    return each_index_array

# Dada una palabra y un diccionario, devuelve un vector representando la palabra 
def token_to_embedding(word, embeddings):
    if word in embeddings.keys():
        return embeddings.get(word)
    else:
        return np.zeros(300)
