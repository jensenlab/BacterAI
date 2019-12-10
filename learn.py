import ast
import copy
import csv 
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sklearn.metrics

import dnf
import neural

def new_batch(rule, predict_net, inputs, K=1000, threshold=0.5):
    # Exhaustive evaluation
    # TODO: Replace with search function

    result = predict_net.predict_probability(inputs)
    # print("results\n", result)
    candidate_indexes = np.where(result > threshold)[0]
    # print("indexes\n", candidate_indexes)
    candidates = np.take(inputs, candidate_indexes, axis=0)
    # print("candidates\n", candidates.shape, "\n", candidates)
    
    candidate_cardinality = np.sum(candidates, axis=1)
    
    # print("candidate_cardinality\n",candidate_cardinality.shape, "\n",  candidate_cardinality)

    min_K_indexes = candidate_cardinality.argsort()[:K]
    # print(min_K_indexes)
    n_found = min_K_indexes.size
    if n_found > 0:
        min_cardinality = candidate_cardinality[min_K_indexes[0]]
    else:
        min_cardinality = None
    # results = np.apply_along_axis(model.model.predict, 1, data)
    # print(results)
    
    n_needed = K - n_found
    if n_needed > 0:
        print(f"Added {n_needed} random indexes")
        all_indexes = np.array(range(inputs.shape[0]))
        valid_indexes = np.setdiff1d(all_indexes, min_K_indexes)
        random_indexes = np.random.choice(valid_indexes, size=(1, n_needed))
        min_K_indexes = np.concatenate([min_K_indexes, random_indexes], 
                                       axis=None)
    
    # print(min_K_indexes)
    return result, min_K_indexes, min_cardinality

def get_metrics(model, data, data_labels):
    predictions = model.predict_class(data)

    precision = sklearn.metrics.precision_score(data_labels, predictions)
    accuracy = sklearn.metrics.accuracy_score(data_labels, predictions)
    recall = sklearn.metrics.recall_score(data_labels, predictions)

    print(f"Precision: {precision}, Accuracy: {accuracy}, Recall: {recall}")
    return precision, accuracy, recall

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # generate all xi
        # length 16 to start 
    generate_new_data = False
    data_filename = "all_data_16.csv"
    rule_filename = "all_data_16_rule.pkl"
    if generate_new_data:
        rule = dnf.Rule(16, poisson_mu_AND=10, poisson_mu_OR=20)
        rule.generate_data_csv("all_data_16", repeat=16)
    else:
        rule = dnf.Rule.from_pickle(rule_filename)
    print("Rule:", rule)
    
    
    max_n = 1000
    data = np.genfromtxt(data_filename, delimiter=",")
    data_x = data[:, :-1]
    data_y = data[:, -1]
    data_dims = data_x.shape
    
    test_indexes = np.random.choice(range(data_dims[0]), 
                                        size=int(0.1*data_dims[0]), 
                                        replace=False)
    test_data_x = data_x[test_indexes, :]
    test_data_y = data_y[test_indexes]
    
    batch_size = 300    
    n_cycles = data_dims[0]//batch_size    
    model = neural.PredictNet()
    #Initialize random
    train_data = np.random.choice([0, 1], size=(batch_size, rule.data_length))
    train_data_labels = np.random.choice([0, 1], size=(batch_size, 1))
    #TEST a set xi
    
    # Initialize graphing
    # Graph 1
    fig = plt.figure(figsize=(5,10))
    x_values = range(n_cycles)
    precision_values = np.empty((n_cycles))
    accuracy_values = np.empty((n_cycles))
    recall_values = np.empty((n_cycles))
    precision_values.fill(np.nan)
    accuracy_values.fill(np.nan)
    recall_values.fill(np.nan)
    ax1 = fig.add_subplot(311)
    line1, = ax1.plot(x_values, precision_values, 'r')
    line2, = ax1.plot(x_values, accuracy_values, 'g')
    line3, = ax1.plot(x_values, recall_values, 'b')
    plt.xlabel(f'Cycle')
    plt.ylabel(f'Metric Value')
    plt.legend(['Precision', 'Accuracy', 'Recall'])
    plt.ylim(0, 1)
    plt.xlim(0, n_cycles)
    
    # Graph 2
    min_cardinality_values = np.empty((n_cycles))
    min_cardinality_values.fill(np.nan)
    ax2 = fig.add_subplot(312)
    line4, = ax2.plot(x_values, min_cardinality_values, 'b')
    plt.xlabel(f'Cycle')
    plt.ylabel(f'Minimum Cardinality')
    plt.ylim(0, 20)
    plt.xlim(0, n_cycles)
    
    # Graph 3
    ax3 = fig.add_subplot(313)
    # line5, = ax3.plot([np.nan], [np.nan], 'b.')
    # line5, _, _, _ = 
    # axs[2].hist2d([0,1], [1,0], range=[[0, 1], [0, 1]], bins=(80, 10), norm=colors.LogNorm())
    plt.xlabel(f'Predicted Value')
    plt.ylabel(f'True Value')
    # plt.ylim(0, 1)
    # plt.xlim(0, 1)
    
    for cycle in range(n_cycles):
        print(f"\nCYCLE {cycle}")
        ## TRAIN g_(x)
        # use MDN? neural net
        # x -> g_ -> P(g(x) = 1)
        model.train(train_data, train_data_labels, epochs=5)
        # accuracy = model.evaluate(test_data_x, test_data_y)[1]
        ## PREDICT
        # get g_(x)
        
        # evaluate function
            # for now, evaluate g_(x) at all xi (brute force)
            # eventually use search algo to explore search space
        # choose batch size K
        # find the K samples with smallest |x| that meet criteria:
            # g_(x) > 0.5
        
        result, new_batch_indexes, min_cardinality = (
            new_batch(rule, model, data_x, K=batch_size, threshold=0.5))
        
        
        # Set new training data after picking new batch
        train_data = data_x[new_batch_indexes, :]
        train_data_labels = data_y[new_batch_indexes]
        
        
        # Remove used experiments from "experiment set"
        data_x = np.delete(data_x, new_batch_indexes, axis=0)
        data_y = np.delete(data_y, new_batch_indexes)
        
        
        ## OUTPUT
            # lowest |x| found
            # Accuracy??: g_(x) > 0.5 + g(x) = 1
            
            # Precision (tp / (tp + fp))
            # Recall (tp / (tp + fn))
            # Accuracy (tp + tn) / (tp + tn + fp + fn)
        precision, accuracy, recall = get_metrics(model, test_data_x, 
                                                  test_data_y)
        precision_values[cycle] = precision
        accuracy_values[cycle] = accuracy
        recall_values[cycle] = recall
        min_cardinality_values[cycle] = min_cardinality
        line1.set_ydata(precision_values)
        line2.set_ydata(accuracy_values)
        line3.set_ydata(recall_values)
        line4.set_ydata(min_cardinality_values)
        
        index_subset = np.random.choice(range(test_data_x.shape[0]), 
                                        size=int(0.1*test_data_x.shape[0]), 
                                        replace=False)
        test_set_predictions = model.predict_probability(test_data_x[index_subset])
        # line5.set_xdata(test_set_predictions)
        # line5.set_ydata(test_data_y[index_subset])
        ax3.cla()
        _, _, _, im = ax3.hist2d(test_set_predictions.flatten(), 
                                                test_data_y[index_subset], 
                                                range=[[0, 1], [0, 1]], 
                                                bins=(80, 10), 
                                                norm=colors.LogNorm())
        if cycle is 0:
            plt.colorbar(im, ax=ax3) 
        plt.pause(0.01)
                        
        
        print(f"ACCURACY: {accuracy}, MIN CARDINALITY: {min_cardinality}")
        cycle += 1
        
        ## REPEAT
    plt.show()
        
    