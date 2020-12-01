# DEPENDENCIAS (Bibliotecas)
# ----------------------------------------------------------------------------------------------------
import warnings
warnings.simplefilter("ignore", UserWarning)

import time
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# DEPENDENCIAS (Locales)
# ----------------------------------------------------------------------------------------------------
import modules.const as const
from modules.vectorizer import Vectorizer

# CLASE PRINCIPAL
# ----------------------------------------------------------------------------------------------------
class Model():

    # Leer dataset de archivo CSV
    def read_dataset(self):

        # Leer dataset en pandas dataframe
        df = pd.read_csv(self.data_path + const.DATA_TRAIN_FILE, sep='|', engine='python', quotechar='"', error_bad_lines=False)

        # Mezclar dataset aleatoriamente
        df = df.sample(frac=1)

        # Guardar dataset en clase, separar ejemplos y categorias
        self.dataframe = df        
        self.dataset = df['texto'].values.astype('U')
        self.categories = df['odio'].values

        # Read testset as Pandas DataFrame
        df_test = pd.read_csv(self.data_path + const.DATA_TEST_FILE, sep='|', engine='python', quotechar='"', error_bad_lines=False)

        # Mezclar testset aleatoriamente
        df_test = df_test.sample(frac=1)

        # Guardar testset en clase, separar ejemplos y categorias
        self.test_dataframe = df_test            
        self.test_dataset = df_test['texto'].values.astype('U')
        self.test_categories = df_test['odio'].values

    # Vectorizar dataset para que el modelo pueda procesarlo
    def vectorize_dataset(self):
        
        # Crear interfaz de vectorizacion
        self.vectorizer = Vectorizer()
        
        # Para vectorizacion de embeddings, procesar dataframe completo y dataset
        tic = time.time()
        self.dataframe = self.vectorizer.transform(self.dataframe)
        self.dataset = list(np.array(self.dataframe['texto'], dtype=object))
        toc = time.time()
        print('(MODEL) Dataset vectorized in ' + '{0:.2f}'.format(toc - tic) + ' seconds')

    # Funcion Auxiliar - Guardar modelo en archivo
    def save(self):
        pickle.dump(self.classifier, open(const.MODEL_FOLDER + const.MODEL_FILE, 'wb'))

    # Funcion Auxiliar - Cargar modelo de archivo
    def load(self):
        self.classifier = pickle.load(open(const.MODEL_FOLDER + const.MODEL_FILE, 'rb'))

    # Constructor
    def __init__(self, model='mlp_classifier', data_path=const.DATA_FOLDER, params={}):

        # Dataset vacio para entrenar
        self.dataset = None
        self.categories = None

        # Testset vacio para evaluar
        self.test_dataset = None
        self.test_categories = None

        # Interfaces auxiliares
        self.dataframe = None        
        self.classifier = None
        self.vectorizer = None

        # Valores de configuracion
        self.model = params['model'] if 'model' in params else model
        self.params = params['params'] if 'params' in params else None
        self.data_path = data_path
    
        # Leer dataset, ejemplos y categorias
        self.read_dataset()

        # Vectorizar dataset y guardar vectorizador
        self.vectorize_dataset()

    # Crear y entrenar clasificador segun modelo elegido
    def train(self):

        # Si los parámetros son pasados al crear modelo, usarlos
        if self.params is not None:
            if self.model == 'svm':
                self.classifier = SVC(**self.params)
            elif self.model == 'tree':
                self.classifier = DecisionTreeClassifier(**self.params)
            elif self.model == 'knn':
                self.classifier = KNeighborsClassifier(**self.params)
            elif self.model == 'mlp_classifier':
                self.classifier = MLPClassifier(**self.params)

        # Si no hay parámetros en el modelo, usar genéricos
        else:
            if self.model == 'svm':
                self.classifier = SVC(kernel='linear', C=10)
            elif self.model == 'tree':
                self.classifier = DecisionTreeClassifier(criterion='entropy', max_depth=8)
            elif self.model == 'knn':
                self.classifier = KNeighborsClassifier(3)
            elif self.model == 'mlp_classifier':
                self.classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', 'alpha'=0.05, learning_rate: 'adaptive', solver='adam',  max_iter=2000)

        # Entrenar usando dataset
        tic = time.time()
        self.classifier.fit(self.dataset, self.categories)
        toc = time.time()
        print('(MODEL) Model trained in ' + '{0:.2f}'.format(toc - tic) + ' seconds')

    # Predecir clasificacion para conjunto X
    def predict(self, X):
        
        # Vectorizar texto
        examples = self.vectorizer.transform(X)
        examples = np.array(examples['texto'], dtype=object)
        examples = list(map(lambda a: np.zeros(300) if len(a) != 300 else a,examples))

        # Generar clasificacion y probabilidades para cada clase
        prediction = self.classifier.predict(examples)

        return prediction

    # Generar evaluacion normal para parametros elegidos
    def evaluate(self):
        prediction = self.predict(self.test_dataframe)

        accuracy = accuracy_score(self.test_categories, prediction)
        results = classification_report(self.test_categories, prediction, output_dict=True)
        report_string = classification_report(self.test_categories, prediction)
        matrix = confusion_matrix(self.test_categories, prediction)

        report = {
            'f1_score': results['macro avg']['f1-score'],
            'precision': results['macro avg']['precision'],
            'recall': results['macro avg']['recall'],
        }

        return accuracy, report, report_string, matrix
