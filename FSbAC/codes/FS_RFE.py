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
from sklearn.feature_selection import RFE
from sys import argv


#fileName = argv[1]
#nfeatures = int(argv[2])
fileName = 'dataConcat'
nfeatures = 29
print("fileName:", fileName)
print("nfeatures:", nfeatures)


imp = SimpleImputer(missing_values=np.nan, strategy='median')

### For the entire dataset
dfTr = pd.read_csv('../data/'+fileName+'.csv', sep =',', header=None, na_values=-1, index_col=None)

idxlabel = dfTr.shape[1] - 1

X = dfTr.iloc[:,:idxlabel]
y = dfTr.iloc[:,idxlabel]

imp.fit(X)
X = imp.transform(X)

model = SGDClassifier()
rfe = RFE(estimator=model, n_features_to_select=nfeatures, step=1)
tic = time.time()
rfe.fit(X, y)
toc = time.time()
FS = np.where(rfe.support_)[0]

np.savetxt('../ResultsOther/FS/'+fileName+'.RFE.txt', FS, fmt='%d')
print("Runtime for the entire data:", toc - tic)
##########################


### For the 10-fold:
nfolds = 10
times = np.zeros(shape=[nfolds,])
for i in range(nfolds):
    print(i)
    dfTr = pd.read_csv('../data/folds/'+fileName+'.train.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    
    idxlabel = dfTr.shape[1] - 1
    
    X = dfTr.iloc[:,:idxlabel]
    y = dfTr.iloc[:,idxlabel]
    
    imp.fit(X)
    X = imp.transform(X)
    
    rfe = RFE(estimator=model, n_features_to_select=nfeatures, step=1)
    tic = time.time()
    rfe.fit(X, y)
    toc = time.time()
    FS = np.where(rfe.support_)[0]
    
    times[i] = toc - tic
    np.savetxt('../ResultsOther/FS/'+fileName+'.RFE.p'+str(i)+'.txt', FS, fmt='%d')
np.savetxt('../ResultsOther/FSruntimes/'+fileName+'.RFE.txt', times, fmt='%.10f')
