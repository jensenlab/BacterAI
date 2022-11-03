import os, sys, re
import pandas as pd
import numpy as np


sys.path.append("../")
import models
import global_vars
from rule_solver_genetic import GeneticSolver

def import_neural_network(path):
    model = models.NeuralNetModel.load_trained_models(path)
    return model

def import_data(path):
    data = pd.read_csv(path, index_col=None)
    return data

def evaluate_rule(rule, data, fitness, ingredients):
    n_ingredients = len(ingredients)
    data['fitness'] = fitness
    data = data.to_numpy()
    rule = np.array(rule)
    choices = np.arange(n_ingredients + 1)
    pop_size = 1000
    n_groups = 4
    solver = GeneticSolver(None, None, n_groups, choices, pop_size, data, None, ingredient_names=ingredients, no_test_train_split=True)
    _, metrics = solver.fitness_score(rule, include_metrics=True, use_test_data=False)
    # _, test_metrics = solver.fitness_score(rule, include_metrics=True, use_test_data=True)
    print('evaluate_neural_network:')
    for k, v in metrics.items():
        print(f'\t{k}={v:.3f}')
    # print('test=')
    # for k, v in test_metrics.items():
    #     print(f'\t{k}={v:.3f}')

def evaluate_neural_network(model, data, true_fitness, threshold):
    predictions, _ = model.evaluate(data, clip=False)
    predictions[predictions >= threshold] = 1
    predictions[predictions < threshold] = 0
    accuracy = (predictions == true_fitness).sum()/len(predictions)

    pred_grow = predictions == 1
    true_grow = true_fitness == 1

    tp = np.sum(np.logical_and(pred_grow, true_grow))/len(pred_grow)
    tn = np.sum(np.logical_and(~pred_grow, ~true_grow))/len(pred_grow)
    fp = np.sum(np.logical_and(pred_grow, ~true_grow))/len(pred_grow)
    fn = np.sum(np.logical_and(~pred_grow, true_grow))/len(pred_grow)
    
    tpr = 0 if (tp + fn) == 0 else tp / (tp + fn)
    tnr = 0 if (tn + fp) == 0 else tn / (tn + fp)
    balanced_accuracy = (tpr + tnr) / 2
    print('evaluate_neural_network:')
    print(f'\tTPR={tpr:.3f}')
    print(f'\tTNR={tnr:.3f}')
    print(f'\t{accuracy=:.3f}')
    print(f'\t{balanced_accuracy=:.3f}')
    return accuracy

def name_to_rule_index(ingredient_name):
    all_ingredients = global_vars.AA_SHORT + global_vars.BASE_NAMES
    return all_ingredients.index(ingredient_name) + 1 # 0 is reserved for no ingredient

def str_to_rule(rule_str):
    rule_str = rule_str.lower()
    pattern = r'\(([a-z ]*)\)'
    matches = re.findall(pattern, rule_str)
    rule = []
    for group in matches:
        ingredients = group.split(' or ')
        ingredients = [name_to_rule_index(i) for i in ingredients]
        needed_to_pad = 4 - len(ingredients)
        ingredients.extend([0] * needed_to_pad)
        rule.extend(ingredients)
    
    print('str_to_rule:')
    print(f'\t{rule_str}')
    print(f'\t{rule}')
    return rule

def main(experiment_path, rule, ingredients, threshold=0.25):
    rule = str_to_rule(rule)
    data_path = os.path.join(experiment_path, "experiment_data.csv")
    data = import_data(data_path)
    data_inputs = data.iloc[:, :len(ingredients)]
    fitness = data.loc[:, 'fitness_median'].copy()
    fitness[fitness >= threshold] = 1
    fitness[fitness < threshold] = 0

    model_path = os.path.join(experiment_path, "final_models")
    model = import_neural_network(model_path)
    model_acc = evaluate_neural_network(model, data_inputs, fitness, threshold)
    
    evaluate_rule(rule, data_inputs, fitness, ingredients)



if __name__ == '__main__':
    EXPERIMENT_PATH = '../published_data/SGO aerobic'
    EXPERIMENT_RULE = '(arg)(leu)(phe)(ser)(tyr)(cys or val)(gln or glu)' #SGO
    # EXPERIMENT_PATH = '../published_data/SSA aerobic'
    # EXPERIMENT_RULE = '(gly)(arg or cys)(cys or leu)(gln or glu)(leu or val)'

    INGREDIENTS = global_vars.AA_SHORT

    main(EXPERIMENT_PATH, EXPERIMENT_RULE, INGREDIENTS)