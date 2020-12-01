# DEPENDENCIAS (Bibliotecas)
# ----------------------------------------------------------------------------------------------------
import sys

# DEPENDENCIAS (Locales)
# ----------------------------------------------------------------------------------------------------
import modules.const as const
from modules.model import Model

# FUNCIONES AUXILIARES
# ----------------------------------------------------------------------------------------------------

# Generar espacio de parámetros según modelo
def get_parameter_space(model):
    
    parameter_space = {}
    
    if model == 'svm':
        parameter_space = [
            {
                'kernel': 'rbf',
                'gamma': 'auto',
                'C': 1
            },
            {
                'kernel': 'rbf',
                'gamma': 'auto',
                'C': 10
            },
            {
                'kernel': 'linear',
                'C': 1
            },
            {
                'kernel': 'linear',
                'C': 10
            }
        ]

    if model == 'tree':
        parameter_space = [
            {
                'criterion': 'gini',
                'max_depth': 4
            },
            {
                'criterion': 'gini',
                'max_depth': 8
            },
            {
                'criterion': 'gini',
                'max_depth': 12
            },
            {
                'criterion': 'entropy',
                'max_depth': 4
            },
            {
                'criterion': 'entropy',
                'max_depth': 8
            },
            {
                'criterion': 'entropy',
                'max_depth': 12
            },
        ]

    if model == 'knn':
        parameter_space = [
            {
                'n_neighbors': 3,
                'weights': 'uniform',
                'metric': 'euclidean'
            },
            {
                'n_neighbors': 5,
                'weights': 'uniform',
                'metric': 'euclidean'
            },
            {
                'n_neighbors': 7,
                'weights': 'uniform',
                'metric': 'euclidean'
            },
            {
                'n_neighbors': 3,
                'weights': 'distance',
                'metric': 'euclidean'
            },
            {
                'n_neighbors': 5,
                'weights': 'distance',
                'metric': 'euclidean'
            },
            {
                'n_neighbors': 7,
                'weights': 'distance',
                'metric': 'euclidean'
            },
            {
                'n_neighbors': 3,
                'weights': 'uniform',
                'metric': 'manhattan'
            },
            {
                'n_neighbors': 5,
                'weights': 'uniform',
                'metric': 'manhattan'
            },
            {
                'n_neighbors': 7,
                'weights': 'uniform',
                'metric': 'manhattan'
            },
            {
                'n_neighbors': 3,
                'weights': 'distance',
                'metric': 'manhattan'
            },
            {
                'n_neighbors': 5,
                'weights': 'distance',
                'metric': 'manhattan'
            },
            {
                'n_neighbors': 7,
                'weights': 'distance',
                'metric': 'manhattan'
            },
        ]

    elif model == 'mlp_classifier':
        parameter_space = [
            {
                'hidden_layer_sizes': (100,),
                'max_iter': 2000,
                'activation': 'relu',
                'solver': 'sgd',
                'alpha': 0.0001,
                'learning_rate': 'constant'
            },
            {
                'hidden_layer_sizes': (100,100),
                'max_iter': 2000,
                'activation': 'relu',
                'solver': 'sgd',
                'alpha': 0.0001,
                'learning_rate': 'constant'
            },
            {
                'hidden_layer_sizes': (50,50,50),
                'max_iter': 2000,
                'activation': 'relu',
                'solver': 'sgd',
                'alpha': 0.0001,
                'learning_rate': 'constant'
            },
            {
                'hidden_layer_sizes': (100,),
                'max_iter': 2000,
                'activation': 'logistic',
                'solver': 'sgd',
                'alpha': 0.0001,
                'learning_rate': 'constant'
            },
            {
                'hidden_layer_sizes': (100,100),
                'max_iter': 2000,
                'activation': 'logistic',
                'solver': 'sgd',
                'alpha': 0.0001,
                'learning_rate': 'constant'
            },
            {
                'hidden_layer_sizes': (50,50,50),
                'max_iter': 2000,
                'activation': 'logistic',
                'solver': 'sgd',
                'alpha': 0.0001,
                'learning_rate': 'constant'
            },
            {
                'hidden_layer_sizes': (100,),
                'max_iter': 2000,
                'activation': 'relu',
                'solver': 'sgd',
                'alpha': 0.05,
                'learning_rate': 'adaptative'
            },
            {
                'hidden_layer_sizes': (100,100),
                'max_iter': 2000,
                'activation': 'relu',
                'solver': 'sgd',
                'alpha': 0.05,
                'learning_rate': 'adaptative'
            },
            {
                'hidden_layer_sizes': (50,50,50),
                'max_iter': 2000,
                'activation': 'relu',
                'solver': 'sgd',
                'alpha': 0.05,
                'learning_rate': 'adaptative'
            },

        ]

    return parameter_space

# FUNCIONES PRINCIPALES
# ----------------------------------------------------------------------------------------------------

# Entrenar un clasificador y evaluar su performance en base a parametros
def evaluate(grid_search=False):

    # Lista de 6-uplas (model, params, accuracy, precision, recall, f1_score)
    results_list = []

    # Iterar segun tipos de modelo
    for model_type in const.MODELS:
            
        print()
        print('(EVALUATOR) Evaluating model ' + model_type)

        if grid_search:

            # Lista de 6-uplas (model, params, accuracy, precision, recall, f1_score)
            grid_search_list = []
            param_space = get_parameter_space(model_type)

            for params in param_space:

                # 1. Crear modelo
                model = Model(model=model_type, params={'model': model_type, 'params': params})

                # 2. Entrenar clasificador
                model.train()

                # 3. Evaluar clasificador
                accuracy, results, _, _ = model.evaluate()
                grid_search_list.append((model_type, params, accuracy, results['precision'], results['recall'], results['f1_score']))

            # Ordenar resultados segun f1_score
            grid_search_list = sorted(grid_search_list, key=lambda x: x[5], reverse=True)

            print()
            print('(EVALUATOR) Grid search results -> Model - ', model_type)
            for _, params, accuracy, precision, recall, f1_score in grid_search_list:
                print()
                print("Params - ", params)
                print("-> F1 Score - ", "{0:.2f}".format(f1_score))
                print("-> Precision - ", "{0:.2f}".format(precision))
                print("-> Recall - ", "{0:.2f}".format(recall))
                print("-> Accuracy - ", "{0:.2f}".format(accuracy))
            print()

            best_params = grid_search_list[0][1]
            best_accuracy = grid_search_list[0][2]
            best_precision = grid_search_list[0][3]
            best_recall = grid_search_list[0][4]
            best_f1_score = grid_search_list[0][5]
            results_list.append((model_type, best_params, best_accuracy, best_precision, best_recall, best_f1_score))

        else:

            # 1. Crear modelo
            model = Model(model=model_type)

            # 2. Entrenar clasificador
            model.train()

            # 3. Evaluar clasificador
            accuracy, results, _, _ = model.evaluate()
            results_list.append((model_type, None, accuracy, results['precision'], results['recall'], results['f1_score']))

    # Ordenar resultados segun f1_score
    results_list = sorted(results_list, key=lambda x: x[5], reverse=True)

    # Mostrar resultados
    print()
    print('(EVALUATOR) Sorted results: ')
    for model, params, accuracy, precision, recall, f1_score in results_list:
        print()
        print("Model - ", model)
        if params is not None:
            print("Params - ", params)
        print("-> F1 Score - ", "{0:.2f}".format(f1_score))
        print("-> Precision - ", "{0:.2f}".format(precision))
        print("-> Recall - ", "{0:.2f}".format(recall))
        print("-> Accuracy - ", "{0:.2f}".format(accuracy))
    print()

    best_solution = {
        'model': results_list[0][0],
        'params': results_list[0][1]
    }

    # Elegir mejor modelo, entrenarlo por completo y guardarlo
    model = Model(model=results_list[0][0], params=best_solution)
    model.train()
    model.save()

    print('(EVALUATOR) Trained and saved best model')

if __name__ == "__main__":

    # 1. Leer parametros (evaluacion superficial o profunda, busqueda de parametros o no)
    main_grid_search = int(sys.argv[1]) == 1 # 0 no, 1 yes

    # 2. Evaluar modelo por defecto en base a parametros
    evaluate(main_grid_search)
