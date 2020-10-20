# DEPENDENCIAS (Bibliotecas)
# ----------------------------------------------------------------------------------------------------
import sys
import util.const as const

# DEPENDENCIAS (Locales)
# ----------------------------------------------------------------------------------------------------
from model import Model

# FUNCIONES PRINCIPALES
# ----------------------------------------------------------------------------------------------------

# Entrenar un clasificador y evaluar su performance en base a parametros
def evaluate(hard_evaluation=False, grid_search=False):

    # Hard evaluation: Evalua un clasificador de cada modelo, los ordena por performance y entrena
    # el de mejor resultados
    if hard_evaluation:

        # Lista de 5-uplas (model, accuracy, precision, recall, f1_score)
        results_list = []

        # Iterar segun tipos de modelo
        for model_type in const.MODELS:
                
            print()
            print('(EVALUATOR) Evaluating model ' + model_type)

            # 1. Crear modelo
            model = Model(model=model_type)

            # 2. Entrenar clasificador
            model.train(grid_search=grid_search)

            # 3. Evaluar clasificador
            accuracy, results, _, _ = model.evaluate()
            results_list.append((model_type, accuracy, results['precision'], results['recall'], results['f1_score']))

        # Ordenar resultador segun f1_score
        results_list = sorted(results_list, key=lambda x: x[4], reverse=True)

        # Mostrar resultados
        print()
        print('(EVALUATOR) Sorted results: ')
        for model, accuracy, precision, recall, f1_score in results_list:
            print()
            print("Model - ", model)      
            print("-> F1 Score - ", "{0:.2f}".format(f1_score))
            print("-> Precision - ", "{0:.2f}".format(precision))
            print("-> Recall - ", "{0:.2f}".format(recall))
            print("-> Accuracy - ", "{0:.2f}".format(accuracy))

        # Elegir mejor modelo, entrenarlo por completo y guardarlo
        model = Model(model=results_list[0][0])
        model.train()
        model.save()

        print()
        print('(EVALUATOR) Trained and saved best model')

    # Soft evaluation: Evalua un clasificador de cada modelo y muestra su performance
    else:

        # Iterar segun tipos de modelo
        for model_type in const.MODELS:

            print()

            # 1. Crear modelo
            model = Model(model=model_type)
            print('(EVALUATOR) Model ' + model_type + ' created')

            # 2. Entrenar clasificador
            model.train(grid_search=main_grid_search)
            print('(EVALUATOR) Model ' + model_type + ' trained')

            # 3. Evaluar clasificador
            accuracy, _, report, matrix = model.evaluate()
            print('(EVALUATOR) Model ' + model_type + ' evaluated')
            print()
            print('(EVALUATOR) Evaluation, Accuracy: ' + str(accuracy))
            print('(EVALUATOR) Evaluation, Classification Report: ')
            print(report)
            print()
            print('(EVALUATOR) Evaluation, Confusion Matrix: ')
            print(matrix)
            print()

if __name__ == "__main__":

    # 1. Leer parametros (evaluacion superficial o profunda, busqueda de parametros o no)
    main_hard_evaluation = int(sys.argv[1]) == 1 # 0 no, 1 yes
    main_grid_search = int(sys.argv[2]) == 1 # 0 no, 1 yes

    # 2. Evaluar modelo por defecto en base a parametros
    evaluate(main_hard_evaluation, main_grid_search)
