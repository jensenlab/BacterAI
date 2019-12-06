import ast
import csv 
import os
import pickle

import numpy as np

import dnf
import neural

def new_batch(rule, predict_net, inputs):
    # Exhaustive evaluation
    # TODO: Replace with search function

    result = predict_net.model.predict(inputs)
    print("results\n", result)
    candidate_indexes = np.where(result > 0.5)[0]
    print("indexes\n", candidate_indexes)
    candidates = np.take(inputs, candidate_indexes, axis=0)
    print("candidates\n", candidates.shape, "\n", candidates)
    
    candidate_cardinality = np.sum(candidates, axis=1)
    
    print("candidate_cardinality\n",candidate_cardinality.shape, "\n",  candidate_cardinality)
    # results = np.apply_along_axis(model.model.predict, 1, data)
    # print(results)
    return None

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    # r = dnf.Rule(16, poissons_mu_AND=5, poisson_mu_OR=12)
    # r.generate_data_csv("all_data_16", repeat=16)
    
    data_filename = "all_data_16.csv"
    rule_filename = "all_data_16_rule.pkl"
    
    data = np.genfromtxt(data_filename, delimiter=",")
    data_x = data[-10:, :-1]
    data_y = data[-10:, -1]
    test_data_x = data_x[:10, :]
    test_data_y = data_y[:10]
    
    rule = dnf.Rule.from_pickle(rule_filename)
    print("Rule:", rule)
    
    batch_size = 1000
    model = neural.PredictNet()
    #Initialize random
    initial_data = np.random.choice([0, 1], size=(batch_size, rule.data_length))
    initial_data_labels = np.random.choice([0, 1], size=(batch_size, 1))
    #TEST a set xi

    model.train(initial_data, initial_data_labels, epochs=5)
    model.evaluate(test_data_x, test_data_y)
    
    batch = new_batch(rule, model, data_x)
    
    
    
    # for x, y in train_dataset:  # only take first element of dataset
    #     model.train(x, y, epochs=5)
    
    # for x, y in test_dataset:
    
    
    
    ## TRAIN g_(x)
    # use MDN? neural net
    # x -> g_ -> P(g(x) = 1)
    
    ## PREDICT
    # get g_(x)
    # generate all xi
        # length 16 to start 
    # evaluate function
        # for now, evaluate g_(x) at all xi (brute force)
        # eventually use search algo to explore search space
    # choose batch size K
    # find the K samples with smallest |x| that meet criteria:
        # g_(x) > 0.5
        
    ## OUTPUT
        # lowest |x| found
        # Accuracy??: g_(x) > 0.5 + g(x) = 1
        # Precision (tp / (tp + fp))
        # Recall (tp / (tp + fn))
        # Accuracy (tp + tn) / (tp + tn + fp + fn```)

    ## REPEAT
        
        
    