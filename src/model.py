# DEPENDENCIAS (Bibliotecas)
# ----------------------------------------------------------------------------------------------------
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# DEPENDENCIAS (Locales)
# ----------------------------------------------------------------------------------------------------
from vectorization.vectorizer import Vectorizer

# DEPENDENCIAS (Locales)
# ----------------------------------------------------------------------------------------------------
import util.const as const

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
        self.vectorizer = Vectorizer(self.vectorization)
        
        # Para vectorizacion de embeddings, procesar dataframe completo y dataset
        if self.vectorization == const.VECTORIZERS['embeddings']:
            self.dataframe = self.vectorizer.fit(self.dataframe)
            self.dataset = list(np.array(self.dataframe['texto'], dtype=object))

        # Para vectorizacion de features, procesar solo dataset
        else:
            self.dataset = self.vectorizer.fit(self.dataset)

    # Funcion Auxiliar - Guardar modelo en archivo
    def save(self):
        pickle.dump(self.classifier, open(const.MODEL_FOLDER + const.MODEL_FILE, 'wb'))

    # Funcion Auxiliar - Cargar modelo de archivo
    def load(self):
        self.classifier = pickle.load(open(const.MODEL_FOLDER + const.MODEL_FILE, 'rb'))

    # Constructor
    def __init__(self, vectorization=const.VECTORIZERS['embeddings'], model='mlp_classifier', data_path=const.DATA_FOLDER, params={}):

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
        self.vectorization = params['vectorization'] if 'vectorization' in params else vectorization
        self.params = params['params'] if 'params' in params else None
        self.data_path = data_path
        
        # Leer dataset, ejemplos y categorias
        self.read_dataset()

        # Vectorizar dataset y guardar vectorizador
        self.vectorize_dataset()

    # Crear y entrenar clasificador segun modelo elegido
    def train(self, grid_search=False):

        # If grid search is setted, train testing each param depending on the chosen model
        if grid_search:
            parameter_space = self.grid_search_evaluate()
            if self.model == 'svm':
                self.classifier = GridSearchCV(SVC(), parameter_space, cv=3, n_jobs=-1)
            elif self.model == 'tree':
                self.classifier = GridSearchCV(DecisionTreeClassifier(), parameter_space, cv=3, n_jobs=-1)
            elif self.model == 'nb':
                self.classifier = GridSearchCV(MultinomialNB(), parameter_space, cv=3, n_jobs=-1)
            elif self.model == 'knn':
                self.classifier = GridSearchCV(KNeighborsClassifier(), parameter_space, cv=3, n_jobs=-1)
            elif self.model == 'mlp_classifier':
                self.classifier = GridSearchCV(MLPClassifier(), parameter_space, cv=3, n_jobs=-1)

        # If params are passed when creating model, train using them
        elif self.params is not None:
            if self.model == 'svm':
                self.classifier = SVC(**self.params)
            elif self.model == 'tree':
                self.classifier = DecisionTreeClassifier(**self.params)
            elif self.model == 'nb':
                self.classifier = MultinomialNB(**self.params)
                self.dataset = self.dataset.todense()
            elif self.model == 'knn':
                self.classifier = KNeighborsClassifier(**self.params)
            elif self.model == 'mlp_classifier':
                self.classifier = MLPClassifier(**self.params)

        # If there are no params and is not a grid search, train using default params
        else:
            if self.model == 'svm':
                self.classifier = SVC(gamma='auto')
            elif self.model == 'tree':
                self.classifier = DecisionTreeClassifier(max_depth=5)
            elif self.model == 'nb':
                self.classifier = MultinomialNB()
                self.dataset = self.dataset.todense()
            elif self.model == 'knn':
                self.classifier = KNeighborsClassifier(5)
            elif self.model == 'mlp_classifier':
                self.classifier = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=2000, solver='sgd')

        # Entrenar usando dataset
        tic = time.time()
        self.classifier.fit(self.dataset, self.categories)
        toc = time.time()
        print('(MODEL) Model trained in ' + '{0:.2f}'.format(toc - tic) + ' seconds')

        # Mostrar mejores hiperparametros para cada modelo
        if grid_search:
            print(f'(MODEL) Best estimator for {self.model}:')
            print(self.classifier.best_estimator_)            
            print('')

            print(f'(MODEL) Best hyper parameters for {self.model}:')
            print(self.classifier.best_params_)
            print('')

            print(f'(MODEL) Best score for {self.model}: (Score: {self.classifier.best_score_})')
            print('')

    # Predecir clasificacion para conjunto X
    def predict(self, X):
        
        # Vectorizar texto
        examples = self.vectorizer.transform(X)

        if self.vectorization == const.VECTORIZERS['embeddings']:
            examples = np.array(examples['texto'], dtype=object)
            examples = list(map(lambda a: np.zeros(300) if len(a) != 300 else a,examples))

        # Generar clasificacion y probabilidades para cada clase
        prediction = self.classifier.predict(examples)

        return prediction

    # Generar evaluacion dependiendo del tipo
    def evaluate(self):
        return self.normal_evaluate()

    # Generar evaluacion normal para parametros elegidos
    def normal_evaluate(self):

        if self.vectorization == const.VECTORIZERS['embeddings']:
            prediction = self.predict(self.test_dataframe)
        else:
            prediction = self.predict(self.test_dataset)

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

    # Generar evaluacion cruzada explorado en espacio de parametros
    def grid_search_evaluate(self):
        parameter_space = {}
        if self.model == 'svm':
            parameter_space = [
                {
                    'kernel': ['rbf'],
                    'gamma': ['auto', 1e-3, 1e-4],
                    'C': [1, 10, 100]
                },
                {
                    'kernel': ['linear'],
                    'C': [1, 10, 100]
                }
            ]
        if self.model == 'tree':
            parameter_space = [
                {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': np.arange(3, 15)
                }
            ]
        if self.model == 'nb':
            parameter_space = [
                {
                    'alpha': [2.0, 1.0, 0.5, 0]
                }
            ]
        if self.model == 'knn':
            parameter_space = [
                {
                    'n_neighbors': [1, 3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            ]
        elif self.model == 'mlp_classifier':
            parameter_space = [
                {
                    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,100), (100,)],
                    'max_iter': [2000],
                    'activation': ['tanh', 'relu', 'logistic'],
                    'solver': ['sgd', 'adam'],
                    'alpha': [0.0001, 0.05],
                    'learning_rate': ['constant', 'adaptive']
                }
            ]
        return parameter_space