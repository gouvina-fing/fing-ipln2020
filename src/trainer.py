# DEPENDENCIES
import os
import sys
from model import Model

# MAIN FUNCTIONS

# Train a model
def train():
    # 1. Create model
    print('(TRAINER) Creating model...')    
    model = Model()

    # 2. Train classifier
    print('(TRAINER) Training model...')
    model.train()

    # 3. Save classifier
    print('(TRAINER) Saving model...')
    model.save()

    return model

if __name__ == "__main__":
    train()
