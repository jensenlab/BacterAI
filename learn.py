
import argparse
import ast
import copy
import csv
import itertools 
import math
import os
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import scipy.stats as sp
from scipy.spatial import distance
import sklearn.metrics

import model
import neural
import utils

parser = argparse.ArgumentParser(description='Run learn.py')
parser.add_argument('-c', '--num_components', required=True, type=int,
                    help='Number of components in media.')
parser.add_argument('-n', '--generate_new_data', action='store_true', default=False,
                    help='Make new data rather than reading from file.')
parser.add_argument('-nm', '--new_model', action='store_true', default=False,
                    help='Generate new model.')
parser.add_argument('-s', '--save_models', action='store_true', default=False,
                    help='Save NN model every cycle.')
args = parser.parse_args()

class Agent():
    def __init__(self, model, predictor, data, num_components):
        self.model = model
        self.predictor = predictor
        self.data = pd.DataFrame(data[1:, :-1])
        self.data_labels = pd.DataFrame(data[1:, -1])
        self.minimum_cardinality = self.model.num_components
        self.current_solution = None
        
        self.data_history = None
        self.data_labels_history = None
    
    def new_batch(self, K=1000, threshold=0.5, use_neural_net=True):
        
        
        def _get_random(data, n):
            random_indexes = random.sample(data.index.to_list(),
                                            min(n, data.index.shape[0]))
            data_random = data.loc[random_indexes, :]
            data = data.drop(random_indexes)
            print("Random Data\n", data_random)
            return data, data_random
        
        def _get_min_K(K, explore_cardinality=None, add_random=False):
            # # Convert to pd.DataFrame
            data = self.data.copy()
            # # Calculate cardinality
            data['cardinality'] = data.sum(axis=1)
            print("Added cardinality\n", data)
            
            if use_neural_net:
                print("USING NEURAL NET")
                # Get probability of grow
                result = self.predictor.predict_probability(
                    data.drop(columns=['cardinality']).to_numpy())
            else:
                print("USING NAIVE BAYES")
                result = self.predictor.predict_bayes(data.drop(
                    columns=['cardinality']).to_numpy())
            
            # Append neural net/naive Bayes predictions
            data['prediction'] = result
            
            if add_random > 0:
                # Added 10% random
                data, data_random = _get_random(data, int(K*add_random))
            
            # Take only data that have a prediction > threshold
            data = data[data.prediction >= threshold] 
                
            print("results\n", result)
            print(f"Num <= 0.50: {result[result <= 0.5].shape[0]}, Num > 0.50: {result[result >= 0.5].shape[0]}")
            
            print("Threshold filter\n", data)
            
            
            
            if explore_cardinality is not None:
                
                target_cardinality = self.minimum_cardinality + explore_cardinality
                data = data[data.cardinality == target_cardinality]
                print("Target card filter\n", data)
                
                # # Sort by cardinality then prediction confidence
                # data = data.sort_values(by=['cardinality', 'prediction_nn'], ascending=[True, False])
                data_tested = self.data_history.copy()
                data_tested['cardinality'] = data_tested.sum(axis=1)
                data_tested = data_tested[data_tested.cardinality == target_cardinality]
                print("Target card filter (history)\n", data_tested)
                
                data_tested = data_tested.drop(columns=['cardinality']).to_numpy()
                
                def _avg_hamming_dist(x):
                    # x is one row of untested data
                    hamming_total = 0
                    if data_tested.shape[0] == 0:
                        return 0
                    for t in data_tested:
                        value = np.count_nonzero(t != x)
                        hamming_total += value
                        # print(value, '  ->  ', t)
                    hamming_avg = hamming_total/data_tested.shape[0]
                    # print("hamming total:", hamming_total, "Avg:", hamming_avg)
                    return hamming_avg
                
                if data.shape[0] > 0:
                    hamming_distances = np.apply_along_axis(_avg_hamming_dist, axis=1, arr=data.to_numpy()[:, :self.model.num_components])
                    
                    data['hamming dist'] = hamming_distances
                    data = data.sort_values(by=['hamming dist', 'cardinality', 'prediction'], ascending=[False, True, False])
                
            else:
                # Take only data that have cardinality < current minimum 
                data = data[data.cardinality < self.minimum_cardinality]
                print("Card filter\n", data)
                
                # Sort by cardinality then prediction confidence
                data = data.sort_values(by=['cardinality', 'prediction'], ascending=[True, False])
            
            if (isinstance(add_random, float) 
                and add_random < 1.0 and add_random > 0.0):
                data = pd.concat([data_random, data])
                print("CONCAT RANDOM", data.shape, data)
            elif add_random >= 1.0:
                data = data_random
                
            current_min_cardinality = data.cardinality.min()
            print("BATCH MIN:", current_min_cardinality)
            
            valid_batch_min_indexes = data[data.cardinality == current_min_cardinality].index.to_numpy()
            print("DATA DataFrame:\n", data)
                
            # Take only the first K indexes        
            min_K_indexes = data.index.to_numpy()[:K]
            
            return min_K_indexes, current_min_cardinality, valid_batch_min_indexes
        

        # Exhaustive evaluation
        # TODO: Replace with search function

        
        min_K_indexes, batch_min_cardinality, valid_batch_min_indexes = _get_min_K(K=K, add_random=0.10)#, explore_cardinality=0)
        # threshold_candidate_indexes, cardinality, all_candidate_indexes, min_K_indexes = _get_min_K()#, explore_cardinality=0)
        
        
        n_needed = K - min_K_indexes.shape[0]
        explore_var = 0
        add_random = 0.0
        stop = False
        while n_needed > 0 and not stop:
            print(f"Exploring ({explore_var})...", f"Number needed: {n_needed}")
            (min_K_indexes_new, batch_min_cardinality_new, 
                valid_batch_min_indexes_new) = (
                    _get_min_K(K=K, explore_cardinality=explore_var, add_random=add_random))
            
            min_K_indexes = np.concatenate([min_K_indexes, min_K_indexes_new], 
                                           axis=None)
            if (batch_min_cardinality_new <= batch_min_cardinality 
                or math.isnan(batch_min_cardinality)):
                batch_min_cardinality = batch_min_cardinality_new
                valid_batch_min_indexes = np.concatenate(
                    [valid_batch_min_indexes, valid_batch_min_indexes_new],
                    axis=None)
                
                
            n_needed = K - min_K_indexes.shape[0]
            explore_var += 1
            if self.minimum_cardinality + explore_var >= self.model.num_components:
                print("Cannot explore any further...choosing random")
                add_random = 1.0
                stop = True
            
            
        # # ununsed_card_cand_indexes = np.setdiff1d(cardinality_candidate_indexes,
        # #                                          min_K_indexes)
        # # ununsed_sorted_cardinality_indexes = np.setdiff1d(sorted_cardinality_indexes,
        # #                                                   min_K_indexes)
        # n_found = min_K_indexes.size
        # # print(results)

        # n_needed = K - n_found
        # if n_needed > 0:
        #     _, _, additional_candidate_indexes, additional_indexes = _get_min_K(result_bayes)[:n_needed]
        #     min_K_indexes = np.concatenate([min_K_indexes, additional_indexes], 
        #                                     axis=None)
        #     print(f"Added {additional_indexes.size} indexes using Bayes")
        #     print("BEFORE", all_candidate_indexes)
        #     print("ADDTL", additional_candidate_indexes)
        #     all_candidate_indexes = np.concatenate([all_candidate_indexes, 
        #                                             additional_candidate_indexes], 
        #                                             axis=None)
        #     print("AFTER ADDING", all_candidate_indexes)
            
            
        # # if n_needed > 0 and add_random:
        # #     # Find random n_needed that have cardinalities less than self.minimum_cardinality
        # #     # if ununsed_sorted_cardinality_indexes.size > 0:
        # #     #     additional_indexes = ununsed_sorted_cardinality_indexes[:n_needed]
        # #     #     ununsed_sorted_cardinality_indexes = (
        # #     #         ununsed_sorted_cardinality_indexes[additional_indexes.size:])
        # #     #     # print(additional_indexes)
        # #     #     print(f"Added {additional_indexes.size} valid cardinality indexes")
                
        # #     # elif ununsed_card_cand_indexes.size > 0:
        # #     #     additional_indexes = ununsed_card_cand_indexes[:n_needed]
        # #     #     ununsed_card_cand_indexes = (
        # #     #         ununsed_card_cand_indexes[additional_indexes.size:])
        # #     #     # print(random_indexes)
        # #     #     min_K_indexes = np.concatenate([min_K_indexes, random_indexes], 
        # #     #                                 axis=None)
        # #     #     print(f"Added {additional_indexes.size} valid threshold indexes")
        # #     # else:
        # #     additional_indexes = np.random.choice(range(result.shape[0]), size=(add_random,))
        # #     print(f"Added {additional_indexes.size} random indexes")
                
            
        # #     min_K_indexes = np.concatenate([min_K_indexes, additional_indexes], 
        # #                                     axis=None)
            
        # #     # n_needed = K - min_K_indexes.size
        
            
        # print(min_K_indexes)
        # batch_min_cardinality = None
        # valid_batch_min_indexes = None
        # if all_candidate_indexes.size > 0:
        #     batch_min_cardinality = (
        #         cardinality[all_candidate_indexes[0]])
        #     # batch_min_index = all_candidate_indexes[0]
        #     all_batch_min_indexes = np.where(cardinality == batch_min_cardinality)[0]
        #     valid_batch_min_indexes = (
        #         np.intersect1d(all_batch_min_indexes, 
        #                     threshold_candidate_indexes))
        # # elif ununsed_sorted_cardinality_indexes.size > 0:
        # #     batch_min_cardinality = (
        # #         cardinality[
        # #             ununsed_sorted_cardinality_indexes[0]])
        
        # min_K_indexes = min_K_indexes.astype(int)
        print("indexes:", min_K_indexes)
        print("Valid batch min:\n", valid_batch_min_indexes)
        
        return min_K_indexes, batch_min_cardinality, valid_batch_min_indexes

    def get_metrics(self, data, data_labels, use_bayes=False):
        if data.size == 0 or data_labels.size == 0:
            print("Cannot calculate metrics... empty data.")
            return None, None, None
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
        L1O_exp = utils.get_LXO(self.model.num_components, 1)
        L2O_exp = utils.get_LXO(self.model.num_components, 2)
        L3O_exp = utils.get_LXO(self.model.num_components, 3)
        
        n = 300
        subset_indexes = random.sample(range(L3O_exp.shape[0]), 
                                       min(n, L3O_exp.shape[0]))
        additional_exp = L3O_exp[subset_indexes]
        # semi random inputs based on L3Os
        # for x in additional_exp:
        #     n = sp.poisson.rvs(1)
        #     candidates_indexes = np.where(x == 1)[0]
        #     set_to_zero = np.random.choice(candidates_indexes, size=n)
        #     x[set_to_zero] = 0
        
        # random_inputs = np.random.choice([0, 1], size=(1000, self.model.num_components))
        
        data = np.concatenate([L1O_exp, L2O_exp, additional_exp])
        data_labels = np.array(self.model.evaluate(data, use_bool=False, 
                                                   use_multiprocessing=False))
        return pd.DataFrame(data), pd.DataFrame(data_labels)
    
    # @classmethod
    def learn(self, save_models=False):
        # generate all xi
            # length 16 to start 
        
        answer = self.model.minimal_components
        
        cardinality = self.data.sum(axis=1)
        minimum_rule_indexes = self.data[cardinality == len(answer)].index
        # cardinality = np.sum(self.data, axis=1)
        # minimum_rule_indexes = np.where(cardinality == answer)[0]
        minimum_rule_data_x = self.data.loc[minimum_rule_indexes, :]
        minimum_rule_data_y = self.data_labels.loc[minimum_rule_indexes]
        
        batch_size = 300    
        n_cycles = self.data.shape[0]//batch_size    
        # self.minimum_cardinality = self.model.num_components
        
        batch_train_data, batch_train_data_labels = self.get_starting_data()
        print(batch_train_data_labels)
        self.data_history = batch_train_data.copy()
        self.data_labels_history = batch_train_data_labels.copy()
        print("batch train data:\n", self.data_history)
        #Random initialization
        # batch_train_data = np.random.choice([0, 1], size=(batch_size, self.model.num_components))
        # batch_train_data_labels = np.random.choice([0, 1], size=(batch_size, 1))
        #TEST a set xi
        
        ############### GRAPHING #################
        # Graph 1
        fig = plt.figure(figsize=(15,12))
        fig.suptitle(f'Target value = {len(answer)}')
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
        plt.ylim(0, self.model.num_components+1)
        plt.xlim(0, n_cycles)
        
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
        plt.ylim(0, self.model.num_components+1)
        plt.xlim(0, n_cycles)
        
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
        plt.title(f'Answer Space (n={len(answer)})')
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
        plt.xlim(0, n_cycles)
        
        # # Graph 8
        grow_proportion = np.zeros((2, n_cycles))
        
        ax8 = fig.add_subplot(424)
        line11 = ax8.stackplot(x_values, grow_proportion, 
                                labels=["Grow", "No Grow"])
        # line12, = ax8.plot(x_values, no_grow_proportion)
        # ax8.fill_between(x_values, 0, grow_proportion, facecolor="#1DACD6", alpha=.7)
        # ax8.fill_between(x_values, grow_proportion, no_grow_proportion, facecolor="#6E5160", alpha=.7)
        plt.title('Grow/No Grow Counts')
        plt.xlabel(f'Cycle')
        plt.ylabel(f'Proportion')
        # plt.legend(['No Grow', 'Grow'])
        plt.ylim(0, 1.1)
        plt.xlim(0, n_cycles)
        already_found = False
        for cycle in range(n_cycles):
            print(f"\nCYCLE {cycle}")
            print(f"Data history:  {self.data_history.shape[0]} {self.data_labels_history.shape[0]}")
            ## TRAIN g_(x)
            # use MDN? neural net
            # x -> g_ -> P(g(x) = 1)
            print(f"Batch train data: \n{batch_train_data}")
            print(f"Batch train data labels: \n{batch_train_data_labels}")
            use_neural_net = True
            if cycle > 0 and batch_train_data.shape[0] > 0:
                _, accuracy, _ = self.get_metrics(
                    batch_train_data.to_numpy(), 
                    batch_train_data_labels.to_numpy())
                _, accuracy_bayes, _ = self.get_metrics(
                    batch_train_data.to_numpy(), 
                    batch_train_data_labels.to_numpy(), 
                    use_bayes=True)
                
                accuracy_NN_pred[cycle] = accuracy
                accuracy_bayes_pred[cycle] = accuracy_bayes
                line9.set_ydata(accuracy_NN_pred)
                line10.set_ydata(accuracy_bayes_pred)
                # use_neural_net = accuracy >= accuracy_bayes
        
            
            # Retrain a new model every time
            self.predictor = type(self.predictor)()
            print('fitting predictors')
            
            # print("XXXXX\n", self.data_history.to_numpy(), '\n', self.data_labels_history.to_numpy())
            self.predictor.train(self.data_history.to_numpy(), 
                                 self.data_labels_history.to_numpy(), epochs=5)
            self.predictor.train_bayes(self.data_history.to_numpy(), 
                                       self.data_labels_history.to_numpy())
            
            if save_models:
                model_name = f"NN_{self.model.num_components}_C{cycle}.h5"
                output_path = os.path.join("models", save_models, model_name)
                self.predictor.model.save(output_path)
            
            ## PREDICT
            # get g_(x)
            
            # evaluate function
                # for now, evaluate g_(x) at all xi (brute force)
                # eventually use search algo to explore search space
            # choose batch size K
            # find the K samples with smallest |x| that meet criteria:
                # g_(x) > 0.5
            
            
            add_random = max(0, n_cycles - (cycle**2))
            
            new_batch_indexes, batch_min_cardinality, batch_min_indexes = (
                self.new_batch(K=batch_size, threshold=0.5, 
                               use_neural_net=use_neural_net))
            
            # if new_batch_indexes.size == 0:
            #     break
            # Set new training data after picking new batch
            batch_train_data = self.data.loc[new_batch_indexes, :]
            batch_train_data_labels = self.data_labels.loc[new_batch_indexes]
            print("BATCH TRAIN DATA", batch_train_data.shape, batch_train_data)
            self.data_history = pd.concat([self.data_history, batch_train_data])
            print("DATA HISTORY", self.data_history.shape, self.data_history)
            
            self.data_labels_history = pd.concat(
                [self.data_labels_history, batch_train_data_labels])
            
            # filter new batch results
            cardinality = batch_train_data.sum(axis=1)
            labels = batch_train_data_labels.rename(columns={0: 'labels'})
            batch = pd.concat([cardinality, labels], axis=1)
            batch.rename(columns={0: 'cardinality'}, inplace=True)
            batch = batch[(batch.cardinality < self.minimum_cardinality) & (batch.labels == 1)]
            print("BATCH\n", batch)
            
            # Evaluate new batch to check for growth & new min solutions
            if batch.size > 0:
                self.minimum_cardinality = batch.cardinality.min()
                minimum_index = (
                    batch[batch.cardinality == 
                          self.minimum_cardinality].index[0])
                self.current_solution = self.data.loc[minimum_index, :]
                print(f"INPUT FOR NEW MIN ({self.minimum_cardinality}): {self.current_solution}")
            
            # for proportion stats
            predictions = self.predictor.predict_class(
                self.data.loc[new_batch_indexes, :].to_numpy())
            
            # Remove used experiments from "experiment set"
            self.data.drop(new_batch_indexes, inplace=True)
            self.data_labels.drop(new_batch_indexes, inplace=True)
            
            #Get a subset of data to get stats
            test_indexes = np.random.choice(self.data.index, 
                                            size=int(0.001*self.data.shape[0]), 
                                            replace=False)

            test_data_x = self.data.loc[test_indexes, :]
            test_data_y = self.data_labels.loc[test_indexes]  

            precision, accuracy, recall = self.get_metrics(
                test_data_x.to_numpy(), test_data_y.to_numpy())
            precision_values[cycle] = precision
            accuracy_values[cycle] = accuracy
            recall_values[cycle] = recall
            
            precision_answer_space, accuracy_answer_space, recall_answer_space = (
                self.get_metrics(minimum_rule_data_x.to_numpy(), 
                                 minimum_rule_data_y.to_numpy()))
            precision_values_answer_space[cycle] = precision_answer_space
            accuracy_values_answer_space[cycle] = accuracy_answer_space
            recall_values_answer_space[cycle] = recall_answer_space

            batch_cardinality_values[cycle] = batch_min_cardinality
            min_cardinality_values[cycle] = self.minimum_cardinality

            
            grow_proportion[0, cycle] = (
                predictions[predictions == 1].shape[0]/batch_size)
            grow_proportion[1, cycle] = (
                predictions[predictions == 0].shape[0]/batch_size)
            ax8.cla()
            line11 = ax8.stackplot(x_values, grow_proportion)
            ax8.legend(["Grow", "No Grow"])
            plt.title('Grow/No Grow Counts')
            plt.xlabel(f'Cycle')
            plt.ylabel(f'Proportion')
            plt.xlim(0, n_cycles)
            
            
            line1.set_ydata(precision_values)
            line2.set_ydata(accuracy_values)
            line3.set_ydata(recall_values)
            line4.set_ydata(batch_cardinality_values)
            line5.set_ydata(min_cardinality_values)
            line6.set_ydata(precision_values_answer_space)
            line7.set_ydata(accuracy_values_answer_space)
            line8.set_ydata(recall_values_answer_space)

        
            if self.minimum_cardinality == len(answer) and not already_found:
                already_found = True
                ax4.annotate(
                    f'Found! ({cycle})',
                    xy=(cycle, len(answer)), xytext=(20, 20),
                    textcoords='offset points', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
            
            if batch_min_cardinality == len(answer):
                ax2.plot([cycle], [len(answer)], 'r+', markersize=6)

            index_subset = np.random.choice(test_data_x.index, 
                                            size=int(0.001*test_data_x.shape[0]), 
                                            replace=False)
           
            test_set_predictions = self.predictor.predict_probability(
                test_data_x.loc[index_subset, :].to_numpy())

            ax3 = fig.add_subplot(428)
            ax3.cla()
            _, _, _, im1 = ax3.hist2d(test_set_predictions.flatten(), 
                                    test_data_y.loc[index_subset].to_numpy().flatten(), 
                                    range=[[0, 1], [0, 1]], 
                                    bins=(80, 10), 
                                    norm=colors.LogNorm())
            plt.xlabel(f'Predicted Value')
            plt.ylabel(f'True Value')
            
            minimum_rule_set_predictions = self.predictor.predict_probability(
                minimum_rule_data_x.to_numpy())
            
            ax6.cla()
            ax6 = fig.add_subplot(427)
            _, _, _, im2 = ax6.hist2d(minimum_rule_set_predictions.flatten(), 
                                    minimum_rule_data_y.to_numpy().flatten(), 
                                    range=[[0, 1], [0, 1]], 
                                    bins=(80, 10), 
                                    norm=colors.LogNorm())
            plt.title(f'Answer Space (n = {len(answer)})')
            plt.xlabel(f'Predicted Value')
            plt.ylabel(f'True Value')
            if cycle is 0:
                plt.colorbar(im1, ax=ax3) 
                plt.colorbar(im2, ax=ax6)
            plt.pause(0.01)
            
            print(f"ACCURACY: {accuracy}, BATCH MIN CARDINALITY: {batch_min_cardinality}, OVERALL MIN CARDINALITY: {self.minimum_cardinality}")
            cycle += 1
            
            ## REPEAT
        print("################ DONE! #################")
        print("CORRECT MINIMUM:", len(answer), "SET:", answer) 
        found_solution = list()
        for idx, c in enumerate(self.current_solution):
            if c == 1:
                found_solution.append(self.model.media_ids[idx])
        
        print("FOUND MINIMUM:", self.minimum_cardinality, "SET:", found_solution)
        plt.show()
        # return minimum_rule
    
    


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    model_folder = "iSMUv01_CDM_2020-01-17T153130538/12"
    
    if args.new_model:
        m = model.Model("models/iSMUv01_CDM.xml", 16)
        model_path = m.make_minimal_media_models(max_n=1)
        data_path = os.path.join("/".join(model_path.split("/")[:-1]), 
                                 f"data_{args.num_components}.csv")
    else:
        folder_path = os.path.join("models", model_folder)
        model_filename = "iSMUv01_CDM_12.xml"
        model_path = os.path.join(folder_path, model_filename)
        data_path = os.path.join(folder_path, f"data_{args.num_components}.csv")
   
    print(model_path)
    
    m = model.Model(model_path, num_components=args.num_components,
                    new_data=args.generate_new_data)

    # print(m.print_compartments())
    
    if args.generate_new_data:
        m.generate_data_csv(use_multiprocessing=True)
        
    print(f"MIN ({len(m.minimal_components)}): {m.minimal_components}")
    # m.print_compartments()
    # with m.model as mod:
    #     import cobra
    #     minimal_medium = cobra.medium.minimal_medium(
    #         mod, m.growth_cutoff, minimize_components=True,
    #         media_components=m.media_ids)
    #     print("\nMINIMAL MEDIA:", minimal_medium.index.to_list())
    #     reaction_ids = [rxn.id for rxn in mod.exchanges]
    #     # reaction_ids = [rxn.id for rxn in mod.reactions if "exch" in rxn.id]
    #     non_media_exchanges = set(reaction_ids) - set(m.media_ids)
    #     print("\nNONMEDIA RXNs:", non_media_exchanges)
    #     components_of_interest = set(minimal_medium.index) - non_media_exchanges
    #     print("\nRXN TO KEEP:", components_of_interest)
        
    #     print('\nBEFORE REMOVAL:', mod.slim_optimize())
    #     reactions = list(set(m.media_ids) - components_of_interest)
    #     print('\nRXN TO REMOVE:', reactions)
    #     reactions_remove = [mod.reactions.get_by_id(rxn) for rxn in reactions]
    #     result, result2 = m.reaction_knockout(reactions_remove, m.growth_cutoff)
    #     print('\nAFTER REMOVAL:', result, result2)
    #     # mod.reactions.get_by_id('NADP_exch').knock_out()
    #     # print('1', mod.slim_optimize())
    # with m.model as mod:
    #     mod.reactions.get_by_id('ribflv_exch').knock_out()
    #     print('2', mod.slim_optimize())
    #     mod.reactions.get_by_id('NADP_exch').knock_out()
    #     print('after', mod.slim_optimize())
        
        
    data = np.genfromtxt(data_path, delimiter=",")
    # data = np.genfromtxt("fours.csv", delimiter=",")
    
    # data = pd.DataFrame(data[1:, :])
    # data["card"] = data.sum(axis=1)
    # data = data[data["card"] < 4].drop(columns=["card"]).to_numpy()
    # np.savetxt("fours.csv", data, delimiter=',', fmt='%i')
    
    # print(data.shape)
    # # # s = data.iloc[:, 0].sum()
    # # # print(s)
    
    # import tensorflow as tf
    # new_model = tf.keras.models.load_model("models/iSMUv01_CDM_2020-01-17T153130538/12/NN_22_C2.h5")
    # # results = new_model.predict_classes(data[1:, :])
    # # print(np.sum(results))
    # results = new_model.evaluate(data[1:, :-1], data[1:, -1], verbose=2)
    # print(results)
    
    predictor = neural.PredictNet()
    agent = Agent(m, predictor, data, args.num_components)
    
    if args.save_models:
        agent.learn(save_models=model_folder)
    else:
        agent.learn()
        
    # install command: pip3 install --upgrade --user ../cobrapy/
    