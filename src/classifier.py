# DEPENDENCIAS (Bibliotecas)
# ----------------------------------------------------------------------------------------------------
import os
import sys

# DEPENDENCIAS (Locales)
# ----------------------------------------------------------------------------------------------------
from model import Model

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
    texts = [text]

    # 2. Predecir categoria
    main_prediction = predict(texts)
