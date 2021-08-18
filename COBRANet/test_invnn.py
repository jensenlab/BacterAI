#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:54:58 2020

@author: jensen
"""

import tensorflow as tf
import numpy as np

from gurobi import GRB

model = tf.keras.models.Sequential()
#model.add(Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu", input_shape=(20,)))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.load_weights('NN_20_C1.h5')

X_test = np.random.randint(0,1+1,size=(10,20))

ynn = model.predict(X_test)

import invnn

gmodel = invnn.NNmilp(model, branch_priority=1)

zmilp = gmodel.predict(X_test)
ymilp = 1 / (1 + np.exp(-zmilp))
print(np.hstack((ymilp.reshape((-1,1)), ynn)))

gmodel.quiet = False
gmodel.set_zL(lb=[0.6])
gmodel.model.setParam(GRB.Param.TimeLimit, 500)
gmodel.model.setParam(GRB.Param.MIPFocus, 1)
gmodel.model.setParam(GRB.Param.MIPGapAbs, 0.9)
gmodel.model.setParam(GRB.Param.ImproveStartTime, 10)
gmodel.model.setParam(GRB.Param.PoolSearchMode, 1)
gmodel.model.setParam(GRB.Param.PoolSolutions, 1000)

for x in gmodel.x.values():
    x.start = 1

gmodel.optimize_x_count()
print(gmodel.model.getAttr(GRB.Attr.SolCount))
print(gmodel.get_x())
