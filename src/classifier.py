# DEPENDENCIAS (Bibliotecas)
# ----------------------------------------------------------------------------------------------------
import os
import sys
import pandas as pd

# DEPENDENCIAS (Locales)
# ----------------------------------------------------------------------------------------------------
from modules.model import Model

# FUNCIONES PRINCIPALES
# ----------------------------------------------------------------------------------------------------

# Dada una lista de ejemplos, predecir su clasificacion en base al modelo por defecto
def predict(examples):

    # 1. Crear modelo
    print('(CLASSIFIER) Creating model...')
    model = Model()

    # 2. Cargar clasificador
    print('(CLASSIFIER) Loading model...')
    model.load()

    # 3. Calcular prediccion
    prediction = model.predict(examples)
    print('(CLASSIFIER) Prediction obtained (' + str(prediction) + ')')

    return prediction

if __name__ == "__main__":

    # 1. Obtener texto de parametro
    text = sys.argv[1]
    texts = pd.DataFrame(columns=['texto'])
    texts = texts.append({ 'texto': text}, ignore_index=True)

    # 2. Predecir categoria
    main_prediction = predict(texts)
