#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:20:39 2022

@author: rve
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv('../data/dataConcat.csv', sep =',', header=None, na_values=-1, index_col=None)
dfD = pd.read_csv('../data/dataConcat.discr.csv', sep =',', header=None, na_values=-1, index_col=None)
labels = np.loadtxt('../data/dataConcat.labels', dtype='int')
minsupsF = pd.read_csv('../data/dataConcat.discr.minsups.csv', sep =',', header=0, index_col=None)

skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=None) #Setting a random_state has no effect since shuffle is False
skf.get_n_splits(dfD, labels)
for i, (train_index, test_index) in enumerate(skf.split(dfD, labels)):
    print(f"Fold {i}:")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")
    Xtrain = dfD.iloc[train_index, :]
    Xtest = dfD.iloc[test_index, :]
    Ytrain = labels[train_index]
    nlabel = np.unique(Ytrain, return_counts=True)
    minsups = np.floor(minsupsF['minsup'] * nlabel[1])
    XOtrain = df.iloc[train_index, :]
    XOtest = df.iloc[test_index, :]
    aux = 'p' + str(i)
    Xtrain.to_csv('../data/folds/dataConcat.discr.train.'+aux+'.csv', sep =',', header=False, index=False, na_rep=-1, float_format='%d')
    Xtrain.to_csv('../data/folds/dataConcat.discr.train.'+aux+'.PM', sep =' ', header=False, index=False, na_rep=999999, float_format='%d', columns=range(Xtrain.shape[1]-1))
    np.savetxt('../data/folds/dataConcat.labels.train.'+aux, Ytrain, fmt='%d')
    minsups.to_csv('../data/folds/dataConcat.discr.minsups.'+aux, sep =' ', header=False, index=True, float_format='%d')
    Xtest.to_csv('../data/folds/dataConcat.discr.test.'+aux+'.csv', sep =',', header=False, index=False, na_rep=-1, float_format='%d')
    XOtrain.to_csv('../data/folds/dataConcat.train.'+aux+'.csv', sep =',', header=False, index=False, na_rep=-1)
    XOtest.to_csv('../data/folds/dataConcat.test.'+aux+'.csv', sep =',', header=False, index=False, na_rep=-1)