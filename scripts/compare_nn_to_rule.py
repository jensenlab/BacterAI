import os, sys, re, tempfile

import pandas as pd
import numpy as np


sys.path.append("../")
import models
import constants
from rule_solver_genetic import GeneticSolver

def import_neural_network(path):
    model = models.NeuralNetModel.load_trained_models(path)
    return model

def compute_k_fold_neural_network_accuracy(data_inputs, fitness, ingredients, k, threshold):
    X, y = data_inputs, fitness

    n_ingredients = len(ingredients)
    indexes = np.arange(len(X))
    np.random.shuffle(indexes)

    split_indexes = np.array_split(indexes, k) # Split into near-equal sizes
    groups = [(X.iloc[idxs, :], y.iloc[idxs]) for idxs in split_indexes]

    accs = []
    for fold in range(k):
        train_groups = [groups[x] for x in range(k) if x != fold]
        test_group = groups[fold]
        X_train = pd.concat([x[0] for x in train_groups], ignore_index=True).to_numpy()
        y_train = pd.concat([x[1] for x in train_groups], ignore_index=True).to_numpy()

        X_test = test_group[0].to_numpy()
        y_test = test_group[1].to_numpy()
        y_test[y_test >= threshold] = 1
        y_test[y_test < threshold] = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            print(temp_dir)
            model = models.NeuralNetModel(temp_dir)
            model.train(
                X_train,
                y_train,
                n_ingredients=n_ingredients,
                n_bags=25,
                bag_proportion=1.0,
                epochs=50,
                batch_size=360,
                lr=0.001,
            )
            acc = evaluate_neural_network(model, X_test, y_test, threshold)
        
        accs.append(acc)
        print(f'Fold {fold+1} - Accuracy: {acc}')
    
    k_fold_acc = sum(accs)/len(accs)
    print(f'{k}-fold accuracy: {k_fold_acc}')
    return k_fold_acc

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
    print('evaluate_rule:')
    for k, v in metrics.items():
        print(f'\t{k}={v:.3f}')

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
    all_ingredients = constants.AA_SHORT + constants.BASE_NAMES
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

    thresholded_fitness = fitness.copy()
    thresholded_fitness[thresholded_fitness >= threshold] = 1
    thresholded_fitness[thresholded_fitness < threshold] = 0

    model_path = os.path.join(experiment_path, "final_models")
    model = import_neural_network(model_path)
    
    k_fold_model_acc = compute_k_fold_neural_network_accuracy(data_inputs, fitness, ingredients, 4, threshold)
    final_model_acc = evaluate_neural_network(model, data_inputs, thresholded_fitness, threshold)
    evaluate_rule(rule, data_inputs, thresholded_fitness, ingredients)


if __name__ == '__main__':
    EXPERIMENT_PATH = '../published_data/SGO aerobic'
    EXPERIMENT_RULE = '(arg)(leu)(phe)(ser)(tyr)(val)(gln or glu)' #SGO
    # EXPERIMENT_PATH = '../published_data/SSA aerobic'
    # EXPERIMENT_RULE = '(gly)(arg or cys)(cys or leu)(gln or glu)(leu or val)' #SSA

    INGREDIENTS = constants.AA_SHORT

    main(EXPERIMENT_PATH, EXPERIMENT_RULE, INGREDIENTS)
