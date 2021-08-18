#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:49:47 2020

@author: jensen
"""

import numpy as np

import gurobi as grb
from gurobi import GRB


class NNmilp():
    def __init__(self, model, intype=GRB.BINARY, branch_priority=0):
        weights = list(zip(model.weights[::2], model.weights[1::2]))
        L = len(weights)
        
        indim = lambda l: weights[l][0].shape[0]
        outdim = lambda l: weights[l][0].shape[1]
        
        m = grb.Model()
        
        a = [[] for _ in range(L)]
        z = [[] for _ in range(L)]
        
        # x = a[0]
        a[0] = m.addVars(indim(0), vtype=intype)
        for var in a[0].values():
            var.branch_priority = branch_priority
        
        for l in range(L):
            z[l] = m.addVars(outdim(l), lb=-GRB.INFINITY, ub=GRB.INFINITY)
            A = np.hstack((weights[l][0].numpy().T, -np.identity(outdim(l))))
            m.addMConstrs(A, a[l].values() + z[l].values(), '=', -weights[l][1])
            
            # add ReLU
            if l+1 < L:
                a[l+1] = m.addVars(outdim(l), lb=-GRB.INFINITY, ub=GRB.INFINITY)
                for k in range(outdim(l)):
                    #m.addConstr(a[l+1][k] == z[l][k])
                    m.addGenConstrPWL(z[l][k], a[l+1][k], [-1, 0, 1], [0, 0, 1])
        
        x_count = m.addVar()
        m.addConstr(grb.quicksum(a[0]) == x_count)
        
        m.update()
        
        self.model = m
        self.a = a
        self.z = z
        self.x = a[0]
        self.zL = z[-1]
        self.x_count = x_count
        
        self.quiet = True
    
    def set_x(self, value=None, lb=None, ub=None):
        if value is not None:
            lb = value
            ub = value
        for j, x in enumerate(self.x.values()):
            if lb is not None:
                x.setAttr(GRB.Attr.LB, lb[j])
            if ub is not None:
                x.setAttr(GRB.Attr.UB, ub[j])
                
    def get_x(self):
        xout = np.zeros((len(self.x.values()),))
        for i, x in enumerate(self.x.values()):
            xout[i] = x.getAttr(GRB.Attr.X)
        return xout
    
    def clear_x(self):
        for j, x in enumerate(self.x.values()):
            x.setAttr(GRB.Attr.LB, 0)
            x.setAttr(GRB.Attr.UB, 1)
            
    def set_zL(self, value=None, lb=None, ub=None):
        if value is not None:
            lb = value
            ub = value
        for j, z in enumerate(self.zL.values()):
            if lb is not None:
                z.setAttr(GRB.Attr.LB, lb[j])
            if ub is not None:
                z.setAttr(GRB.Attr.UB, ub[j])
                
    def get_zL(self):
        zout = np.zeros((len(self.zL.values()),))
        for i, z in enumerate(self.zL.values()):
            zout[i] = z.getAttr(GRB.Attr.X)
        return zout
                
    def clear_zL(self):
        for j, z in enumerate(self.zL.values()):
            z.setAttr(GRB.Attr.LB, -GRB.INFINITY)
            z.setAttr(GRB.Attr.UB, GRB.INFINITY)
    
    def set_x_count(self, value=None, lb=0, ub=None):
        if value is not None:
            lb = value
            ub = value
        if ub is None:
            ub = len(self.x)
        self.x_count.setAttr(GRB.Attr.LB, lb)
        self.x_count.setAttr(GRB.Attr.UB, ub)
        
    def clear_x_count(self):
        self.set_x_count()
    
    def optimize_zL(self, sense=GRB.MAXIMIZE):
        self.model.setObjective(grb.quicksum(self.zL), sense)
        self.optimize()
    
    def optimize_x_count(self, sense=GRB.MINIMIZE):
        self.model.setObjective(self.x_count, sense)
        self.optimize()
        
    def clear_objective(self):
        self.model.setObjective(0)
        
    def optimize(self):
        if self.quiet:
            self.model.setParam(GRB.Param.OutputFlag, 0)
        else:
            self.model.setParam(GRB.Param.OutputFlag, 1)
        self.model.optimize()
    
    def predict(self, X):
        n = X.shape[0]
        z_pred = np.zeros((n,))
        
        for i in range(n):
            self.set_x(X[i,])
            self.optimize()
            z_pred[i] = self.get_zL()
        self.clear_x()
        return z_pred
    

if __name__ == "__main__":
    from keras.models import Sequential
    from keras.layers import Dense

    dim_in = 4
    
    model = Sequential()
    
    model.add(Dense(units=4, activation='relu', input_shape=(dim_in,)))
    model.add(Dense(units=2, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    train_X = np.random.randint(0,1+1,size=(100,dim_in))
    train_y = (train_X.sum(1) > 3)
    
    model.fit(train_X, train_y, epochs=10)
    gmodel = NNmilp(model)    

    zL = gmodel.predict(train_X)
    ypred = 1 / (1 + np.exp(-zL))
    print(np.hstack((ypred.reshape((-1,1)), model.predict(train_X))))
