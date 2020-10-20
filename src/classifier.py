# DEPENDENCIES
import os
import sys
from model import Model

# MAIN FUNCTIONS

# Given a list of examples, predict its classification using default model
def predict(examples):

    # 1. Create model
    print('(CLASSIFIER) Creating model...')
    model = Model()

    # 2. Load classifier
    print('(CLASSIFIER) Loading model...')
    model.load()

    # 3. Make prediction
    prediction = model.predict(examples)
    print('(CLASSIFIER) Prediction obtained (' + str(prediction) + ')')

    return prediction

if __name__ == "__main__":

    # 1. Get text from args
    text = sys.argv[1]
    texts = [text]

    # 2. Predict category
    main_prediction = predict(texts)
