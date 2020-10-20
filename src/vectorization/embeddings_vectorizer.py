# DEPENDENCIES
import sys
import pandas as pd
import numpy as np
import statistics
import util.const as const
import re
from tokenizer import tokenize

# MAIN FUNCTIONS

# Given a list of tweets and an embedding dictionary,
# returns a list of vectors representing each tweet
def get_vectors(tweets, dictionary):
    tweets['text'] = tweets['text'].apply(convert_tweet_to_embedding, embeddings=dictionary)
    return tweets

# Return word embeddings as dictionary
def get_dictionary():
    dicc = {}

    # Read file as Pandas DataFrame
    df_test = pd.read_csv(const.DATA_FOLDER + const.EMBEDDINGS_FILE, engine='python', sep='\s+', header=None) 

    # Get words and embeddings values
    rows = df_test.shape[0] - 1
    words = df_test.loc[0:rows, 0]
    df = df_test.loc[0:rows, 1:300]

    # Replace NaN and inf with 0
    df[df==np.inf] = np.nan
    df.fillna(0)

    # Fill dicc with every word
    for index, row in df.iterrows():
        if not(words[index] in dicc.keys()):
            dicc[words[index]] = row.values
    
    return dicc

# AUX FUNCTIONS

# Given a tweet and an embedding dictionary, return an embedding vector representing the tweet
def convert_tweet_to_embedding(tweet, embeddings):
    words = np.array(tokenize_text(tweet), dtype=object)
    return mean_of_tweet_embedding(words, embeddings)

# Given a text, tokenize it returning a list of words
def tokenize_text(text):

    # Erase non alphanumerical characters
    regex = r"¡|!|,|\?|\.|=|\+|-|_|&|\^|%|$|#|@|\(|\)|`|'|<|>|/|:|;|\*|$|¿|\[|\]|\{|\}|~"
    text = re.sub(regex, ' ', text)

    # Tokenize
    words = [ token.txt for token in tokenize(text) if token.txt is not None]
    return words

# Given a list of words and an embedding dictionary, return an embedding vector representing the list of words
def mean_of_tweet_embedding(words, embeddings):
    data = pd.Series(words)
    data = data.apply(token_to_embedding, embeddings=embeddings)
    each_index_array = list(zip(*data))
    each_index_array = list(map(statistics.mean,each_index_array))
    each_index_array = np.array(each_index_array)

    return each_index_array

# Given a word and an embedding dictionary, return an embedding vector representing the word 
def token_to_embedding(word, embeddings):
    if word in embeddings.keys():
        return embeddings.get(word)
    else:
        return np.zeros(300)
