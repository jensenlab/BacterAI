import csv
import math
import os
import random

import numpy as np
import scipy.stats as sp
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

import model

class PredictNet():
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
            ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        
        self.model_bayes = GaussianNB()
    
    def train(self, data, data_labels, epochs):
        self.model.fit(data, data_labels, epochs=epochs)
    
    def train_bayes(self, data, data_labels):
        self.model_bayes.fit(data, data_labels)
    
    def predict_bayes(self, data):
        return self.model_bayes.predict(data)
        
    def evaluate(self, data, data_labels):
        return self.model.evaluate(data, data_labels, verbose=2)
    
    def predict_probability(self, data):
        return self.model.predict_proba(data)
    
    def predict_class(self, data):
        return self.model.predict_classes(data)
                    

def generate_training_data():
    m = model.load_cobra("models/iSMUv01_CDM_LOO_v2.xml")
    max_n = 10000
    with open('CDM_leave_out_validation_01.csv', mode='a') as file:
        writer = csv.writer(file, delimiter=',')
        for _ in trange(max_n):
            # n = random.randint(0, len(model.KO_RXN_IDS))
            n = sp.poisson.rvs(5)
            grow, reactions = model.knockout_and_simulate(
                m, n, return_boolean=True)
            reactions = list(reactions)
            reactions.append(grow)
            writer.writerow(reactions)
            # print(grow, reactions)
        
        for _ in trange(max_n):
            n = random.randint(0, len(model.KO_RXN_IDS))
            grow, reactions = model.knockout_and_simulate(
                m, n, return_boolean=True)                
            reactions = list(reactions)
            reactions.append(grow)
            writer.writerow(reactions)
            # print(grow, reactions)
   
   
def load_data(mode='train', max_n=None):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: data and the corresponding labels
    """
    print(f"\nMode: {mode}")
    if mode == 'train':
        raw_train = np.genfromtxt('CDM_leave_out_training_01.csv', delimiter=',')
        raw_validation = np.genfromtxt('CDM_leave_out_validation_01.csv', delimiter=',')
        
        data_train = raw_train[:max_n,:-1] if max_n else raw_train[:,:-1]
        data_validation = raw_validation[:max_n,:-1] if max_n else raw_validation[:,:-1]
        
        data_train_labels = raw_train[:max_n,-1] if max_n else raw_train[:,-1]
        data_validation_labels = raw_validation[:max_n,-1] if max_n else raw_validation[:,-1]
        
        x_train, y_train, x_valid, y_valid = data_train, data_train_labels, \
                                             data_validation, data_validation_labels
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        raw_test = np.genfromtxt('CDM_leave_out_test_01.csv', delimiter=',')
        data_test = raw_test[:max_n,:-1] if max_n else raw_test[:,:-1]
        data_test_labels = raw_test[:max_n,-1] if max_n else raw_test[:,-1]
        
        x_test, y_test = data_test, data_test_labels
        return x_test, y_test
        

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    x_train, y_train, x_valid, y_valid = load_data(max_n=10000)
    x_test, y_test = load_data(mode='test')
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(1000)
    # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    
    model = PredictNet()
    
    for x, y in train_dataset:  # only take first element of dataset
        model.train(x, y, epochs=5)
        predictions = model.model.predict_proba(x_test)
        print(predictions)
        
    
    # for x, y in test_dataset:
    model.evaluate(x_test, y_test)
    
    results = model.predict(x_test)
    print(results)
    

    # generate_training_data()
    
    # scaler = MinMaxScaler()
    # max_n = 1000
    # data = np.genfromtxt('CDM_leave_out_training.csv', delimiter=',')
    # data_train = scaler.fit_transform(data[:max_n, :])
    # print(data_train)
    # x = data_train[:max_n, :-1]
    # y = np.array([[v] for v in data_train[:max_n, -1].tolist()])
    
    # print(y)

    # net = NeuralNetwork(x, y)
    # for _ in trange(100):
    #     net.feed_forward()
    #     net.back_propogation()
        
    # print(net.output)