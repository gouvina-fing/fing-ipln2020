# DEPENDENCIES
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
from vectorization.vectorizer import Vectorizer
import util.const as const

# MAIN CLASS
class Model():
    '''
    Model representation
    '''

    # Create dataset in csv format
    def read_dataset(self):

        # Read dataset as Pandas DataFrame
        df = pd.read_csv(self.data_path + const.DATA_TRAIN_FILE)

        # Shuffle dataset before spliting columns
        df = df.sample(frac=1)

        # Save dataframe, get train dataset and train categories
        self.dataframe = df        
        self.dataset = df['text'].values.astype('U')
        self.categories = df['humor'].values

        # Read dataset as Pandas DataFrame
        df_test = pd.read_csv(self.data_path + const.DATA_TEST_FILE)

        # Shuffle dataset before spliting columns
        df_test = df_test.sample(frac=1)

        # Save dataframe, get train dataset and train categories
        self.test_dataframe = df_test            
        self.test_dataset = df_test['text'].values.astype('U')
        self.test_categories = df_test['humor'].values

    # Vectorize texts for input to model
    def vectorize_dataset(self):
        
        # Create vectorizer interface
        self.vectorizer = Vectorizer(self.vectorization)
        
        # If vectorization type is embeddings, vectorize using dataframe and then extract
        if self.vectorization == const.VECTORIZERS['word_embeddings']:
            self.dataframe = self.vectorizer.fit(self.dataframe)
            self.dataset = list(np.array(self.dataframe['text'], dtype=object))
        
        # If not, vectorize numpy array
        else:
            self.dataset = self.vectorizer.fit(self.dataset)

    # Aux function - For saving classifier
    def save(self):
        pickle.dump(self.classifier, open(const.MODEL_FOLDER + const.MODEL_FILE, 'wb'))

    # Aux function - For loading classifier
    def load(self):
        self.classifier = pickle.load(open(const.MODEL_FOLDER + const.MODEL_FILE, 'rb'))

    # Constructor
    def __init__(self, vectorization=const.VECTORIZERS['features'], model='mlp_classifier', data_path=const.DATA_FOLDER, params={}):

        # Create empty dataset for training
        self.dataset = None
        self.categories = None

        # Create empty testset for evaluation
        self.test_dataset = None
        self.test_categories = None

        # Create other empty objects
        self.dataframe = None        
        self.classifier = None
        self.vectorizer = None

        # Create other configuration values
        self.model = params['model'] if 'model' in params else model
        self.vectorization = params['vectorization'] if 'vectorization' in params else vectorization
        self.params = params['params'] if 'params' in params else None
        self.data_path = data_path
        
        # Read dataset and categories
        self.read_dataset()

        # Vectorize dataset and save vectorizer
        self.vectorize_dataset()

    # Create and train classifier depending on chosen model
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

        # Train using dataset
        tic = time.time()
        self.classifier.fit(self.dataset, self.categories)
        toc = time.time()
        print('(MODEL) Model trained in ' + '{0:.2f}'.format(toc - tic) + ' seconds')

        # Show best hyper parameters for the model
        if grid_search:
            print(f'(MODEL) Best estimator for {self.model}:')
            print(self.classifier.best_estimator_)            
            print('')

            print(f'(MODEL) Best hyper parameters for {self.model}:')
            print(self.classifier.best_params_)
            print('')

            print(f'(MODEL) Best score for {self.model}: (Score: {self.classifier.best_score_})')
            print('')

    # Predict classification for X using classifier
    def predict(self, X):
        
        # Vectorize text
        examples = self.vectorizer.transform(X)

        if self.vectorization == const.VECTORIZERS['word_embeddings']:
            examples = np.array(examples['text'], dtype=object)
            examples = list(map(lambda a: np.zeros(300) if len(a) != 300 else a,examples))

        # Generate classification and probabilities for every class
        prediction = self.classifier.predict(examples)

        return prediction

    # Generate evaluation depending of type
    def evaluate(self):
        return self.normal_evaluate()

    # Generate normal evaluation
    def normal_evaluate(self):

        if self.vectorization == const.VECTORIZERS['word_embeddings']:
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

    # Generate parameter space for model
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