#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 11:56:31 2023

@author: rve
"""

# https://scikit-learn.org/stable/modules/feature_selection.html

import time
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sys import argv


#fileName = argv[1]
#nfeatures = int(argv[2])
fileName = 'dataConcatWoEB.discr'
nfeatures = 34
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

sel = SelectKBest(mutual_info_classif, k=nfeatures)
tic = time.time()
sel.fit(X, y)
toc = time.time()
FS = sel.get_support(indices=True)

np.savetxt('../ResultsOther/FS/'+fileName+'.KBestMI.txt', FS, fmt='%d')
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
    
    sel = SelectKBest(mutual_info_classif, k=nfeatures)
    tic = time.time()
    sel.fit(X, y)
    toc = time.time()
    FS = sel.get_support(indices=True)
    
    times[i] = toc - tic
    np.savetxt('../ResultsOther/FS/'+fileName+'.KBestMI.p'+str(i)+'.txt', FS, fmt='%d')
np.savetxt('../ResultsOther/FSruntimes/'+fileName+'.KBestMI.txt', times, fmt='%.10f')
