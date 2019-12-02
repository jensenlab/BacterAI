import csv
import random
import math

import model

import numpy as np
import scipy.stats as sp
from tqdm import trange 


import numpy as np

# def sigmoid(x):
#     return 1.0/(1+ np.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1.0 - x)

# class NeuralNetwork:
#     def __init__(self, x, y):
#         self.input      = x
#         self.weights1   = np.random.rand(self.input.shape[1],4) 
#         self.weights2   = np.random.rand(4,1)                 
#         self.y          = y
#         self.output     = np.zeros(self.y.shape)

#     def feedforward(self):
#         self.layer1 = sigmoid(np.dot(self.input, self.weights1))
#         self.output = sigmoid(np.dot(self.layer1, self.weights2))

#     def backprop(self):
#         # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
#         d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
#         d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

#         # update the weights with the derivative (slope) of the loss function
#         self.weights1 += d_weights1
#         self.weights2 += d_weights2


# if __name__ == "__main__":
#     X = np.array([[0,0,1],
#                   [0,1,1],
#                   [1,0,1],
#                   [1,1,1]])
#     y = np.array([[0],[1],[1],[0]])
#     nn = NeuralNetwork(X,y)

#     for i in range(1500):
#         nn.feedforward()
#         nn.backprop()

#     print(nn.output)



class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], self.input.shape[0]) 
        self.weights2 = np.random.rand(self.input.shape[0], 1)                 
        self.y = y
        self.output = np.zeros(self.y.shape)

    # def sigmoid(x):
    #     def _f(x):
    #         return 1/(1+math.exp(-x))
    #     f = np.vectorize(_f) 
    #     return f(x)
      
    # def sigmoid_derivative(x):
    #     def _f(x):
    #         return math.exp(-x)/((1+math.exp(-x))**2)
    #     f = np.vectorize(_f) 
    #     return f(x)
    def sigmoid(x):
        return 1.0/(1 + np.exp(-x))

    def sigmoid_derivative(x):
        return x * (1.0 - x)
    
    def feed_forward(self):
        self.layer1 = NeuralNetwork.sigmoid(np.dot(self.input, self.weights1))
        self.output = NeuralNetwork.sigmoid(np.dot(self.layer1, self.weights2))

    def back_propogation(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * NeuralNetwork.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * NeuralNetwork.sigmoid_derivative(self.output), self.weights2.T) * NeuralNetwork.sigmoid_derivative(self.layer1)))

        # print(0, self.layer1.T.shape, (2*(self.y - self.output) * NeuralNetwork.sigmoid_derivative(self.output)).shape)
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        # d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * NeuralNetwork.sigmoid_derivative(self.output)))
        # temp = np.dot(2*(self.y - self.output) * NeuralNetwork.sigmoid_derivative(self.output), self.weights2.T)
        # # print(temp)
        # d_weights1 = np.dot(self.input.T,  (temp * NeuralNetwork.sigmoid_derivative(self.layer1)))
        # print(2, (2*(self.y - self.output) * NeuralNetwork.sigmoid_derivative(self.output)).shape, self.weights2.shape)
        
        # print(1, self.input.T.shape, (np.dot(2*(self.y - self.output) * NeuralNetwork.sigmoid_derivative(self.output), 
                                    # self.weights2) * NeuralNetwork.sigmoid_derivative(self.layer1)).shape)
        
        # d_weights1 = np.dot(self.input.T,  
        #                     (np.dot(2* (self.y - self.output) * NeuralNetwork.sigmoid_derivative(self.output), 
        #                             self.weights2) * NeuralNetwork.sigmoid_derivative(self.layer1)))
        
        # update the weights with the derivative (slope) of the loss function
        # print(3, self.weights1.shape, d_weights1.shape)
        # print(4, self.weights2.shape, d_weights2.shape)
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        
        
def generate_training_data():
    m = model.load_cobra("models/iSMUv01_CDM_LOO_v2.xml")
    with open('CDM_leave_out_test.csv', mode='a') as file:
        writer = csv.writer(file, delimiter=',')
        for _ in trange(1000):
            # n = random.randint(0, len(model.KO_RXN_IDS))
            n = sp.poisson.rvs(5)
            grow, reactions = model.knockout_and_simulate(m, n)
            reactions = list(reactions)
            reactions.append(grow)
            writer.writerow(reactions)
            # print(grow, reactions)
        
        for _ in trange(1000):
            n = random.randint(0, len(model.KO_RXN_IDS))
            grow, reactions = model.knockout_and_simulate(m, n)
            reactions = list(reactions)
            reactions.append(grow)
            writer.writerow(reactions)
            # print(grow, reactions)
            
if __name__ == "__main__":
    # generate_training_data()
    
    # data = np.genfromtxt('CDM_leave_out_training.csv', delimiter=',')
    # x = data[:10000, :-1]
    # # y = np.reshape(data[:100, -1], (1, -1))
    # y = np.array([[v] for v in data[:10000, -1].tolist()])
    # # print(y)

    # net = NeuralNetwork(x, y)
    # for _ in trange(1000):
    #     net.feed_forward()
    #     net.back_propogation()
        
    # print(net.output)