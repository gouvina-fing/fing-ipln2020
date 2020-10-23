# DEPENDENCIAS (Bibliotecas)
# ---------------------------------------------------------------
import os

# CONSTANTES
# ---------------------------------------------------------------

MAIN_ROUTE = str(os.getcwd()).replace('/src', '')

# DIRECTORIOS Y ARCHIVOS
# ---------------------------------------------------------------

DATA_FOLDER = MAIN_ROUTE + "/data"
DATA_TRAIN_FILE = "/train.csv"
DATA_TEST_FILE = "/test.csv"
DATA_VAL_FILE = "/val.csv"
EMBEDDINGS_FILE = '/fasttext.es.300.txt'

MODEL_FOLDER = MAIN_ROUTE + "/models"
MODEL_FILE = "/model.sav"

# TIPOS
# ---------------------------------------------------------------

MODELS = ['svm', 'tree', 'nb', 'knn', 'mlp_classifier']
VECTORIZERS = {
    'features': 1,
    'embeddings': 2,
}