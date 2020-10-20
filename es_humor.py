# python3 es_humor.py <data_path> test_file1.csv … test_fileN.csv
# Donde <data_path> tiene la ruta donde se encuentran los datos impartidos:
#   • humor_train.csv, humor_val.csv, humor_test.csv
#   • intropln2019_embeddings_es_300.txt
#   y test_file1.csv … test_fileN.csv son un conjunto de archivos de test (pueden tener salida o no).

# EJEMPLO DE INVOCACION:
# python3 es_humor.py data/ data_test.csv

# El programa debe:
# 1. Entrenar un clasificador (hiperaparámetros encontrados previamente) utilizando los conjuntos de train y val (debe demorar menos de 10 minutos en una CPU intel i7)
# 2. Por cada archivo test_file1.csv … test_fileN.csv el programa debe:
#    1. aplicar el modelo previamente entrenado
#    2. generar un archivo de salida test_file1.out … test_fileN.out con las salidas obtenidas a cada archivo de test.
#       El archivo de salida debe tener las salidas (0 o 1) en orden y separados por un fin de línea. (Ej. 1\n0\n0\n1...\n0)

import sys, os
sys.path.append(f'{os.getcwd()}/src/')

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import util.const as const
import trainer as trainer
from model import Model

best_solution = {
    'vectorization': const.VECTORIZERS['features'],
    'model': 'mlp_classifier',
    'params': {
        'activation': 'relu',
        'alpha': 0.0001,
        'hidden_layer_sizes': (50, 100, 50),
        'learning_rate': 'adaptive',
        'max_iter': 2000,
        'solver': 'adam'
    }
}

def read_input():
    if len(sys.argv) < 2:
        raise Exception('Cantidad insuficiente de parametros')
    data_path = sys.argv[1]
    test_files = []
    for test_file in sys.argv[2:]:
        test_files.append(test_file)
    return data_path, test_files

def run_test(model, path, test_file):
    ### READ DATA
    df_test = pd.read_csv(path + test_file)
    
    ### PREDICT
    predictions = model.predict(df_test['text'].values.astype('U'))

    ### PROCESS METRICS
    accuracy = accuracy_score(df_test['humor'].values, predictions)
    results = classification_report(df_test['humor'].values, predictions, output_dict=True)
    matrix = confusion_matrix(df_test['humor'].values, predictions)

    report = {
        'f1_score': results['macro avg']['f1-score'],
        'precision': results['macro avg']['precision'],
        'recall': results['macro avg']['recall'],
    }
    print()
    print('(EVALUATION) Accuracy: ' + str(accuracy))
    print('(EVALUATION) Classification Report: ')
    print('--> Precision (avg): ' + "{0:.2f}".format(report['f1_score']))
    print('--> Recall (avg): ' + "{0:.2f}".format(report['recall']))
    print('--> F1 Score (avg): ' + "{0:.2f}".format(report['precision']))
    print('(EVALUATION) Confusion Matrix: ')
    print(matrix)
    print()

    ### SAVE RESULTS
    f = open(f"{test_file.replace('.csv', '')}.out", "a")
    for pred in predictions[:-1]:
        f.write(f"{pred}\n")
    f.write(f"{predictions[-1]}")
    f.close()

def main():
    data_path, test_files = read_input()

    model = Model(data_path=data_path, params=best_solution)

    model.train()

    for test_file in test_files:
        run_test(model, data_path, test_file)
  
main()
