# DEPENDENCIAS (Bibliotecas)
# ----------------------------------------------------------------------------------------------------
import os
import sys

# DEPENDENCIAS (Locales)
# ----------------------------------------------------------------------------------------------------
from modules.model import Model

# FUNCIONES PRINCIPALES
# ----------------------------------------------------------------------------------------------------

# Entrenar un clasificador
def train():

    # 1. Crear modelo
    print('(TRAINER) Creating model...')    
    model = Model()

    # 2. Entrenar clasificador
    print('(TRAINER) Training model...')
    model.train()

    # 3. Guardar clasificador
    print('(TRAINER) Saving model...')
    model.save()

    return model

if __name__ == "__main__":
    train()
