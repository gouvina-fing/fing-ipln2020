# FIRMA DEL PROGRAMA:
# ----------------------------------------------------------------------------------------------------
# python es_odio.py <data_path> test_file1.csv … test_fileN.csv
# Donde <data_path> tiene la ruta donde se encuentran los datos impartidos:
#   • train.csv, val.csv, test.csv
#   • fasttext.es.300.txt
#   y test_file1.csv … test_fileN.csv son un conjunto de archivos de test (pueden tener salida o no).
# EJEMPLO:
# python3 es_odio.py data/ test.csv

# DEPENDENCIAS (Bibliotecas)
# ----------------------------------------------------------------------------------------------------
import sys, os
sys.path.append(f'{os.getcwd()}/src/') # Para poder separar el código en carpeta ./src
import csv
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# DEPENDENCIAS (Locales)
# ----------------------------------------------------------------------------------------------------
import src.modules.const as const
from src.modules.model import Model
import src.trainer as trainer

# Parametros de solucion a entrenar
best_solution = {
    'model': 'mlp_classifier',
    'params': {
        'activation': 'relu',
        'alpha': 0.05,
        'hidden_layer_sizes': (200,),
        'learning_rate': 'adaptive',
        'max_iter': 2000,
        'solver': 'adam'
    }
}

# FUNCIONES
# ----------------------------------------------------------------------------------------------------

# Lectura de parametros
def read_input():
    if len(sys.argv) < 2:
        raise Exception('Cantidad insuficiente de parametros')
    data_path = sys.argv[1]
    test_files = []
    for test_file in sys.argv[2:]:
        test_files.append(test_file)
    return data_path, test_files

# Corrida de test con un modelo
def run_test(model, path, test_file):
    
    # Lectura de conjunto de test
    df_test = pd.read_csv(path + test_file, sep='\t', engine='python', quotechar='"', header=None, error_bad_lines=False)
    df_test.columns = ['texto', 'odio']

    # Ejecucion de modelo sobre conjunto de test
    predictions = model.predict(df_test)

    # Generacion y muestra de metricas
    accuracy = accuracy_score(df_test['odio'].values, predictions)
    results = classification_report(df_test['odio'].values, predictions, output_dict=True)
    matrix = confusion_matrix(df_test['odio'].values, predictions)
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

    # Guardado de resultados en archivo de salida
    f = open(f"{test_file.replace('.csv', '')}.out", "w")
    for pred in predictions[:-1]:
        f.write(f"{pred}\n")
    f.write(f"{predictions[-1]}")
    f.close()

# Entrenamiento y ejecucion de modelo optimo en conjuntos de test
def main():

    # Lectura de parametros
    data_path, test_files = read_input()

    # Creacion y entrenamiento de modelo
    model = Model(data_path=data_path, params=best_solution)
    model.train()
    model.save()
    
    # Ejecucion de modelo sobre cada archivo de test
    for test_file in test_files:
        run_test(model, data_path, test_file)
  
main()
