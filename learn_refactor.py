
import argparse
import ast
import copy
import csv
import itertools 
import os
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.stats as sp
import sklearn.metrics

import dnf
import neural
import utils

parser = argparse.ArgumentParser(description='Run learn.py')
parser.add_argument('-n', '--new_rule', action='store_true',
                    help='Make new rule rather than reading from file.')
args = parser.parse_args()

class Agent():
    def __init__(self, rule, predictor, data):
        self.rule = rule
        self.predictor = predictor
        self.data = data[:, :-1]
        self.data_labels = data[:, -1]
        self.minimum_cardinality = 14 #rule.data_length
        
        self.data_history = None
        self.data_labels_history = None
    
    def new_batch(self, K=1000, threshold=0.5):
    
        def _get_min_K(result, explore_cardinality=None):
            # Get indexes that have result above a threshold
            threshold_candidate_indexes = np.where(result > threshold)[0] #OG INDEXES
            print("threshold_candidate_indexes\n", threshold_candidate_indexes)
            
            # Compute cardinality of inputs
            cardinality = np.sum(self.data, axis=1).astype(int)
            print("candidate_cardinality\n",cardinality.shape, "\n",  cardinality)
            
            if explore_cardinality is not None:
                # Get indexes of cardinalities equal to the current min + exploration parameter
                cardinality_candidate_indexes = np.where(
                    cardinality == self.minimum_cardinality + explore_cardinality)[0]
                print("cardinality_candidate_indexes\n",
                    cardinality_candidate_indexes.shape, "\n",  
                    cardinality_candidate_indexes)
                # Get corresponding input array    
                array_candidates = self.data[cardinality_candidate_indexes, :]
                print("array candidates", array_candidates)
                
                # Get cardinality candidates indexes in order of decreasing 
                # Hamming distance to incumbant solution
                print("Incumbant", self.minimum_cardinality)
                
                
                
                
                # Compute cardinality of already tested 
                cardinality_tested = np.sum(self.data_history, axis=1).astype(int)      
                # Get indexes of already tested cardinalities equal to the current min + exploration parameter
                cardinality_candidate_indexes_tested = np.where(
                    cardinality_tested == self.minimum_cardinality + explore_cardinality)[0]
                print("cardinality_tested\n",cardinality_tested.shape, "\n",  cardinality_tested)
                # Get corresponding input array    
                tested_candidates = self.data_history[cardinality_candidate_indexes_tested, :]
                print("tested_candidates", tested_candidates)
            
            
            
            
            
            
                print("inputs", self.data)
                def _avg_hamming_dist(x):
                    # x is one row of untested data
                    hamming_total = 0
                    for t in tested_candidates:
                        value = np.count_nonzero(t != x)
                        hamming_total += value
                        # print(value, '  ->  ', t)
                    hamming_avg = hamming_total/tested_candidates.shape[0]
                    # print("hamming total:", hamming_total, "Avg:", hamming_avg)
                    return hamming_avg
                
                
                # Get cardinality candidates indexes in order of increasing avg hamming dist     
                hamming_distances = np.apply_along_axis(_avg_hamming_dist, axis=1, arr=array_candidates)
                print("hamming distance", hamming_distances)
                sorted_hamming_indexes = np.argsort(hamming_distances)
                print("sorted_hamming_indexes\n", 
                    sorted_hamming_indexes.shape, "\n",  
                    sorted_hamming_indexes)
                # Reorder cardinality candidate indexes by candidate length 
                cardinality_candidate_indexes = (
                    cardinality_candidate_indexes[sorted_hamming_indexes])
                
            else:
                # Get indexes of cardinalities less than current min
                cardinality_candidate_indexes = np.where(cardinality < self.minimum_cardinality)[0]
                
                # Get cardinality values less than current min    
                cardinality_candidates = cardinality[cardinality_candidate_indexes]
                
                # Get cardinality candidates indexes in order of increasing cardinality     
                sorted_cardinality_indexes = cardinality_candidates.argsort()
                
                # Reorder cardinality candidate indexes by candidate length 
                cardinality_candidate_indexes = (
                    cardinality_candidate_indexes[sorted_cardinality_indexes])
                
                print("sorted_cardinality_indexes\n", sorted_cardinality_indexes.shape, "\n",  sorted_cardinality_indexes)
            
            # All valid candidate indexes after growth threshold and cardinality filters
            all_candidate_indexes = np.array([el for el in 
                                            cardinality_candidate_indexes 
                                            if el in threshold_candidate_indexes]
                                            ).astype(int)
            print("all_candidate_indexes\n",all_candidate_indexes.shape, "\n",  all_candidate_indexes)
            
            
            
            # Take only the first K indexes        
            min_K_indexes = all_candidate_indexes[:K]
            print("min_K_indexes\n",min_K_indexes.shape, "\n",  min_K_indexes)
            
            return threshold_candidate_indexes, cardinality, all_candidate_indexes, min_K_indexes
        

        # Exhaustive evaluation
        # TODO: Replace with search function

        # Get probability of grow
        result = self.predictor.predict_probability(self.data)
        print("results\n", result)
        
        result_bayes = self.predictor.predict_bayes(self.data)
        print("results bayes\n", result_bayes)
        
        
        print(f"Num <= 0.50: {result[result <= 0.5].shape[0]}, Num > 0.50: {result[result >= 0.5].shape[0]}")
        
        threshold_candidate_indexes, cardinality, all_candidate_indexes, min_K_indexes = _get_min_K(result)#, explore_cardinality=0)
        
        # ununsed_card_cand_indexes = np.setdiff1d(cardinality_candidate_indexes,
        #                                          min_K_indexes)
        # ununsed_sorted_cardinality_indexes = np.setdiff1d(sorted_cardinality_indexes,
        #                                                   min_K_indexes)
        n_found = min_K_indexes.size
        # print(results)

        n_needed = K - n_found
        if n_needed > 0:
            _, _, additional_candidate_indexes, additional_indexes = _get_min_K(result_bayes)[:n_needed]
            min_K_indexes = np.concatenate([min_K_indexes, additional_indexes], 
                                            axis=None)
            print(f"Added {additional_indexes.size} indexes using Bayes")
            print("BEFORE", all_candidate_indexes)
            print("ADDTL", additional_candidate_indexes)
            all_candidate_indexes = np.concatenate([all_candidate_indexes, 
                                                    additional_candidate_indexes], 
                                                    axis=None)
            print("AFTER ADDING", all_candidate_indexes)
            
            
        # if n_needed > 0 and add_random:
        #     # Find random n_needed that have cardinalities less than self.minimum_cardinality
        #     # if ununsed_sorted_cardinality_indexes.size > 0:
        #     #     additional_indexes = ununsed_sorted_cardinality_indexes[:n_needed]
        #     #     ununsed_sorted_cardinality_indexes = (
        #     #         ununsed_sorted_cardinality_indexes[additional_indexes.size:])
        #     #     # print(additional_indexes)
        #     #     print(f"Added {additional_indexes.size} valid cardinality indexes")
                
        #     # elif ununsed_card_cand_indexes.size > 0:
        #     #     additional_indexes = ununsed_card_cand_indexes[:n_needed]
        #     #     ununsed_card_cand_indexes = (
        #     #         ununsed_card_cand_indexes[additional_indexes.size:])
        #     #     # print(random_indexes)
        #     #     min_K_indexes = np.concatenate([min_K_indexes, random_indexes], 
        #     #                                 axis=None)
        #     #     print(f"Added {additional_indexes.size} valid threshold indexes")
        #     # else:
        #     additional_indexes = np.random.choice(range(result.shape[0]), size=(add_random,))
        #     print(f"Added {additional_indexes.size} random indexes")
                
            
        #     min_K_indexes = np.concatenate([min_K_indexes, additional_indexes], 
        #                                     axis=None)
            
        #     # n_needed = K - min_K_indexes.size
        
            
        print(min_K_indexes)
        batch_min_cardinality = None
        valid_batch_min_indexes = None
        if all_candidate_indexes.size > 0:
            batch_min_cardinality = (
                cardinality[all_candidate_indexes[0]])
            # batch_min_index = all_candidate_indexes[0]
            all_batch_min_indexes = np.where(cardinality == batch_min_cardinality)[0]
            valid_batch_min_indexes = (
                np.intersect1d(all_batch_min_indexes, 
                            threshold_candidate_indexes))
        # elif ununsed_sorted_cardinality_indexes.size > 0:
        #     batch_min_cardinality = (
        #         cardinality[
        #             ununsed_sorted_cardinality_indexes[0]])
        
        min_K_indexes = min_K_indexes.astype(int)
        return result, min_K_indexes, batch_min_cardinality, valid_batch_min_indexes

    def get_metrics(self, data, data_labels, use_bayes=False):
        if use_bayes:
            predictions = self.predictor.predict_bayes(data)
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
        else:
            predictions = self.predictor.predict_class(data)

        precision = sklearn.metrics.precision_score(data_labels, predictions)
        accuracy = sklearn.metrics.accuracy_score(data_labels, predictions)
        recall = sklearn.metrics.recall_score(data_labels, predictions) 

        print(f"Precision: {precision}, Accuracy: {accuracy}, Recall: {recall}")
        return precision, accuracy, recall


    def get_starting_data(self):
        #Initialize with L1O and L2O data
        L1O_exp = utils.get_LXO(self.rule.data_length, 1)
        L2O_exp = utils.get_LXO(self.rule.data_length, 2)
        L3O_exp = utils.get_LXO(self.rule.data_length, 3)
        
        n = 300
        subset_indexes = random.sample(range(L3O_exp.shape[0]), n)
        additional_exp = L3O_exp[subset_indexes]
        for x in additional_exp:
            n = sp.poisson.rvs(1)
            candidates_indexes = np.where(x == 1)[0]
            set_to_zero = np.random.choice(candidates_indexes, size=n)
            x[set_to_zero] = 0
        
        # random_inputs = np.random.choice([0, 1], size=(1000, self.rule.data_length))
        
        data = np.concatenate([L1O_exp, L2O_exp, additional_exp])
        data_labels = np.array(self.rule.evaluate(data, use_bool=False))
        return data, data_labels
    
    # @classmethod
    def learn(self):
        # generate all xi
            # length 16 to start 
        
        minimum_rule = self.rule.minimum_rule()
        
        cardinality = np.sum(self.data, axis=1)
        minimum_rule_indexes = np.where(cardinality == minimum_rule)[0]
        minimum_rule_data_x = np.take(self.data, minimum_rule_indexes, axis=0)
        minimum_rule_data_y = np.take(self.data_labels, minimum_rule_indexes, axis=0)
        
        batch_size = 300    
        n_cycles = self.data.shape[0]//batch_size    
        # self.minimum_cardinality = self.rule.data_length
        
        batch_train_data, batch_train_data_labels = self.get_starting_data()
        print(batch_train_data_labels)
        self.data_history = np.copy(batch_train_data)
        self.data_labels_history = np.copy(batch_train_data_labels)
        print("batch train data:\n", self.data_history)
        #Random initialization
        # batch_train_data = np.random.choice([0, 1], size=(batch_size, self.rule.data_length))
        # batch_train_data_labels = np.random.choice([0, 1], size=(batch_size, 1))
        #TEST a set xi
        
        ############### GRAPHING #################
        # Graph 1
        fig = plt.figure(figsize=(15,12))
        fig.suptitle(f'Target value = {minimum_rule}')
        fig.subplots_adjust(hspace=.35)
        
        x_values = range(n_cycles)
        precision_values = np.empty((n_cycles))
        accuracy_values = np.empty((n_cycles))
        recall_values = np.empty((n_cycles))
        precision_values.fill(np.nan)
        accuracy_values.fill(np.nan)
        recall_values.fill(np.nan)
        ax1 = fig.add_subplot(421)
        line1, = ax1.plot(x_values, precision_values, 'r')
        line2, = ax1.plot(x_values, accuracy_values, 'g')
        line3, = ax1.plot(x_values, recall_values, 'b')
        plt.title(f'All Training Data (10% subset)')
        plt.xlabel(f'Cycle')
        plt.ylabel(f'Metric Value')
        plt.legend(['Precision', 'Accuracy', 'Recall'])
        plt.ylim(0, 1.1)
        plt.xlim(0, n_cycles)
        
        # Graph 2
        batch_cardinality_values = np.empty((n_cycles))
        batch_cardinality_values.fill(np.nan)
        ax2 = fig.add_subplot(425)
        line4, = ax2.plot(x_values, batch_cardinality_values, 'b')
        plt.title(f'Batch Minimum Cardinality')
        plt.xlabel(f'Cycle')
        plt.ylabel(f'Cardinality')
        plt.ylim(0, self.rule.data_length+1)
        plt.xlim(0, n_cycles+1)
        
        # Graph 3
        ax3 = fig.add_subplot(428)
        plt.xlabel(f'Predicted Value')
        plt.ylabel(f'True Value')
        
        # Graph 4
        min_cardinality_values = np.empty((n_cycles))
        min_cardinality_values.fill(np.nan)
        ax4 = fig.add_subplot(426)
        line5, = ax4.plot(x_values, min_cardinality_values, 'b')
        plt.title(f'Overall Minimum Cardinality')
        plt.xlabel(f'Cycle')
        plt.ylabel(f'Cardinality')
        plt.ylim(0, self.rule.data_length+1)
        plt.xlim(0, n_cycles+1)
        
        # Graph 5
        precision_values_answer_space = np.empty((n_cycles))
        accuracy_values_answer_space = np.empty((n_cycles))
        recall_values_answer_space = np.empty((n_cycles))
        precision_values_answer_space.fill(np.nan)
        accuracy_values_answer_space.fill(np.nan)
        recall_values_answer_space.fill(np.nan)
        ax5 = fig.add_subplot(422)
        line6, = ax5.plot(x_values, precision_values_answer_space, 'r')
        line7, = ax5.plot(x_values, accuracy_values_answer_space, 'g')
        line8, = ax5.plot(x_values, recall_values_answer_space, 'b')
        plt.title(f'Answer Space (n={minimum_rule})')
        plt.xlabel(f'Cycle')
        plt.ylabel(f'Metric Value')
        plt.legend(['Precision', 'Accuracy', 'Recall'])
        plt.ylim(0, 1.1)
        plt.xlim(0, n_cycles)
        
        # Graph 6
        ax6 = fig.add_subplot(427)
        
        # Graph 7
        accuracy_NN_pred = np.empty((n_cycles))
        accuracy_bayes_pred = np.empty((n_cycles))
        accuracy_NN_pred.fill(np.nan)
        accuracy_bayes_pred.fill(np.nan)
        
        ax7 = fig.add_subplot(423)
        line9, = ax7.plot(x_values, accuracy_NN_pred, 'b')
        line10, = ax7.plot(x_values, accuracy_bayes_pred, 'r')
        plt.title('Predictive Scores (NN vs. Bayes)')
        plt.xlabel(f'Cycle')
        plt.ylabel(f'Metric Value')
        plt.legend(['Accuracy (NN)', 'Accuracy (Bayes)'])
        plt.ylim(0, 1.1)
        plt.xlim(0, n_cycles+1)
        
        # # Graph 8
        # precision_values_pred = np.empty((n_cycles))
        # accuracy_values_pred = np.empty((n_cycles))
        # recall_values_pred = np.empty((n_cycles))
        # precision_values_pred.fill(np.nan)
        # accuracy_values_pred.fill(np.nan)
        # recall_values_pred.fill(np.nan)
        # ax8 = fig.add_subplot(424)
        # line12, = ax8.plot(x_values, precision_values_pred, 'r')
        # line13, = ax8.plot(x_values, accuracy_values_pred, 'g')
        # line14, = ax8.plot(x_values, recall_values_pred, 'b')
        # plt.title('Bayes vs. NN')
        # plt.xlabel(f'Cycle')
        # plt.ylabel(f'Metric Value')
        # plt.legend(['Accuracy (Bayes)', 'Accuracy (NN)'])
        # plt.ylim(0, 1.1)
        # plt.xlim(0, n_cycles+1)
        
        for cycle in range(n_cycles):
            print(f"\nCYCLE {cycle}")
            print(f"Data history:  {self.data_history.shape[0]} {self.data_labels_history.shape[0]}")
            ## TRAIN g_(x)
            # use MDN? neural net
            # x -> g_ -> P(g(x) = 1)
            print(f"Batch train data: \n{batch_train_data}")
            print(f"Batch train data labels: \n{batch_train_data_labels}")
            if cycle > 0 and batch_train_data.shape[0] > 0:
                _, accuracy, _ = self.get_metrics(
                    batch_train_data, batch_train_data_labels)
                _, accuracy_bayes, _ = self.get_metrics(
                    batch_train_data, batch_train_data_labels, 
                    use_bayes=True)
                
                accuracy_NN_pred[cycle] = accuracy
                accuracy_bayes_pred[cycle] = accuracy_bayes
                line9.set_ydata(accuracy_NN_pred)
                line10.set_ydata(accuracy_bayes_pred)
        
            
            # Retrain a new model every time
            self.predictor = type(self.predictor)()
            print('fitting predictors')
            self.predictor.train(self.data_history, self.data_labels_history, epochs=5)
            self.predictor.train_bayes(self.data_history, self.data_labels_history)
            
            
            ## PREDICT
            # get g_(x)
            
            # evaluate function
                # for now, evaluate g_(x) at all xi (brute force)
                # eventually use search algo to explore search space
            # choose batch size K
            # find the K samples with smallest |x| that meet criteria:
                # g_(x) > 0.5
                
            add_random = max(0, n_cycles - (cycle**2))
            
            result, new_batch_indexes, batch_min_cardinality, batch_min_indexes = (
                self.new_batch(K=batch_size, threshold=0.5))
            
            # if new_batch_indexes.size == 0:
            #     break
            # Set new training data after picking new batch
            batch_train_data = self.data[new_batch_indexes, :]
            batch_train_data_labels = self.data_labels[new_batch_indexes]
            
            self.data_history = np.concatenate((self.data_history, batch_train_data))
            self.data_labels_history = np.concatenate(
                (self.data_labels_history, batch_train_data_labels))
            
            if batch_min_cardinality is not None:
                for idx in batch_min_indexes:
                    if (self.data_labels[idx] == 1 and 
                        batch_min_cardinality < self.minimum_cardinality):
                        self.minimum_cardinality = batch_min_cardinality
                        print(f"INPUT FOR NEW MIN: {self.data[idx]}")
            
            # Remove used experiments from "experiment set"
            self.data = np.delete(self.data, new_batch_indexes, axis=0)
            self.data_labels = np.delete(self.data_labels, new_batch_indexes)
            
            #Get a subset of data to get stats
            data_dims = self.data.shape
            test_indexes = np.random.choice(range(data_dims[0]), 
                                            size=int(0.1*data_dims[0]), 
                                            replace=False)
            test_data_x = self.data[test_indexes, :]
            test_data_y = self.data_labels[test_indexes]
        
            precision, accuracy, recall = self.get_metrics(test_data_x, 
                                                           test_data_y)
            precision_values[cycle] = precision
            accuracy_values[cycle] = accuracy
            recall_values[cycle] = recall
            
            precision_answer_space, accuracy_answer_space, recall_answer_space = (
                self.get_metrics(minimum_rule_data_x, minimum_rule_data_y))
            precision_values_answer_space[cycle] = precision_answer_space
            accuracy_values_answer_space[cycle] = accuracy_answer_space
            recall_values_answer_space[cycle] = recall_answer_space
            
            
            batch_cardinality_values[cycle] = batch_min_cardinality
            min_cardinality_values[cycle] = self.minimum_cardinality
            
            line1.set_ydata(precision_values)
            line2.set_ydata(accuracy_values)
            line3.set_ydata(recall_values)
            line4.set_ydata(batch_cardinality_values)
            line5.set_ydata(min_cardinality_values)
            line6.set_ydata(precision_values_answer_space)
            line7.set_ydata(accuracy_values_answer_space)
            line8.set_ydata(recall_values_answer_space)
            
            index_subset = np.random.choice(range(test_data_x.shape[0]), 
                                            size=int(0.1*test_data_x.shape[0]), 
                                            replace=False)
            test_set_predictions = self.predictor.predict_probability(test_data_x[index_subset])
            ax3.cla()
            _, _, _, im1 = ax3.hist2d(test_set_predictions.flatten(), 
                                    test_data_y[index_subset], 
                                    range=[[0, 1], [0, 1]], 
                                    bins=(80, 10), 
                                    norm=colors.LogNorm())
            # plt.xlabel(f'Predicted Value')
            # plt.ylabel(f'True Value')
            minimum_rule_set_predictions = self.predictor.predict_probability(minimum_rule_data_x)
            ax6.cla()
            _, _, _, im2 = ax6.hist2d(minimum_rule_set_predictions.flatten(), 
                                    minimum_rule_data_y, 
                                    range=[[0, 1], [0, 1]], 
                                    bins=(80, 10), 
                                    norm=colors.LogNorm())
            # plt.xlabel(f'Predicted Value (for answer space)')
            # plt.ylabel(f'True Value (for answer space)')
            if cycle is 0:
                plt.colorbar(im1, ax=ax3) 
                plt.colorbar(im2, ax=ax6) 
            plt.pause(0.01)
                            
            
            print(f"ACCURACY: {accuracy}, BATCH MIN CARDINALITY: {batch_min_cardinality}, OVERALL MIN CARDINALITY: {self.minimum_cardinality}")
            cycle += 1
            
            ## REPEAT
        print("CORRECT MINIMUM:", minimum_rule, "FOUND:", self.minimum_cardinality)
        # plt.show()
        return minimum_rule
    
    


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    generate_new_data = args.new_rule
    data_filename = "all_data_16.csv"
    rule_filename = "all_data_16_rule.pkl"
    if generate_new_data:
        if os.path.exists(data_filename):
            os.remove(data_filename)
        if os.path.exists(rule_filename):
            os.remove(rule_filename)
        rule = dnf.Rule(16, poisson_mu_AND=8, poisson_mu_OR=14)
        rule.generate_data_csv("all_data_16", repeat=16)
    else:
        rule = dnf.Rule.from_pickle(rule_filename)
    print("Rule:", rule)
    data = np.genfromtxt(data_filename, delimiter=",")
    predictor = neural.PredictNet()
    agent = Agent(rule, predictor, data)
    
    true, found = agent.learn()
