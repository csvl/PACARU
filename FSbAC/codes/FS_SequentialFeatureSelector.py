#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:40:50 2023

@author: rve
"""

# https://scikit-learn.org/stable/modules/feature_selection.html

import time
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sys import argv


fileName = argv[1]
nfeatures = int(argv[2])
direction = argv[3]
# fileName = 'dataConcat'
# nfeatures = 29
# direction="forward"
print("fileName:", fileName)
print("nfeatures:", nfeatures)
print("direction:", direction)


imp = SimpleImputer(missing_values=np.nan, strategy='median')

### For the entire dataset
dfTr = pd.read_csv('../data/'+fileName+'.csv', sep =',', header=None, na_values=-1, index_col=None)

idxlabel = dfTr.shape[1] - 1

X = dfTr.iloc[:,:idxlabel]
y = dfTr.iloc[:,idxlabel]

imp.fit(X)
X = imp.transform(X)

model = SGDClassifier(max_iter=10000)
sfs = SequentialFeatureSelector(model, n_features_to_select=nfeatures, direction=direction)
tic = time.time()
sfs.fit(X, y)
toc = time.time()
FS = sfs.get_support(indices=True)

np.savetxt('../ResultsOther/FS/'+fileName+'.SFS'+direction+'.txt', FS, fmt='%d')
print("Runtime for the entire data:", toc - tic)
##########################


### For the 10-fold:
nfolds = 10
times = np.zeros(shape=[nfolds,])
for i in range(nfolds):
    dfTr = pd.read_csv('../data/folds/'+fileName+'.train.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    
    idxlabel = dfTr.shape[1] - 1
    
    X = dfTr.iloc[:,:idxlabel]
    y = dfTr.iloc[:,idxlabel]
    
    imp.fit(X)
    X = imp.transform(X)
    
    model = SGDClassifier(max_iter=2000)
    sfs = SequentialFeatureSelector(model, n_features_to_select=nfeatures, direction=direction)
    tic = time.time()
    sfs.fit(X, y)
    toc = time.time()
    FS = sfs.get_support(indices=True)
    
    times[i] = toc - tic
    np.savetxt('../ResultsOther/FS/'+fileName+'.SFS'+direction+'.p'+str(i)+'.txt', FS, fmt='%d')
np.savetxt('../ResultsOther/FSruntimes/'+fileName+'.SFS'+direction+'.txt', times, fmt='%.10f')
