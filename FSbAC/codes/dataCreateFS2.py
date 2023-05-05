#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:47:06 2023

@author: rve
"""
import pandas as pd
import numpy as np

### DATA CONSIDERING ONLY THE SELECTED FEATURES


fileInfoAttr = 'attrInfo'
fileName = 'dataConcat'
woeb = '' # '' or 'WoEB'
discr = '' # '' or '.discr'
alg = 'SFSforward'

infoAttr = pd.read_csv('../data/'+fileInfoAttr+woeb+'.csv', sep =',', header=0, index_col=None)
idxlabel = infoAttr.shape[0]


FS = np.loadtxt('../ResultsOther/FS/'+fileName+woeb+discr+'.'+alg+'.txt', dtype=int)
FSWLabel = FS.copy()
FSWLabel = np.append(FSWLabel, idxlabel)

infoAttrNew = infoAttr.iloc[FS,:]

# Printing public data
infoAttrNew.to_csv('../data/'+fileInfoAttr+woeb+alg+'.csv', sep =',', header=True, index=False)
for i in range(10): # Loop in the 10-folds
    Tr = pd.read_csv('../data/folds/'+fileName+woeb+'.train.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    Te = pd.read_csv('../data/folds/'+fileName+woeb+'.test.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    TrD = pd.read_csv('../data/folds/'+fileName+woeb+'.discr.train.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    TeD = pd.read_csv('../data/folds/'+fileName+woeb+'.discr.test.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    
    FS = np.loadtxt('../ResultsOther/FS/'+fileName+woeb+discr+'.'+alg+'.p'+str(i)+'.txt', dtype=int)
    FSWLabel = FS.copy()
    FSWLabel = np.append(FSWLabel, idxlabel)
    
    infoAttrNew = infoAttr.iloc[FS,:]
    
    infoAttrNew.to_csv('../data/folds/'+fileInfoAttr+woeb+alg+'.p'+str(i)+'.csv', sep =',', header=True, index=False)
    Tr.to_csv('../data/folds/'+fileName+woeb+alg+'.train.p'+str(i)+'.csv', sep =',', header=False, index=False, na_rep=-1, columns=FSWLabel)
    TrD.to_csv('../data/folds/'+fileName+woeb+alg+'.discr.train.p'+str(i)+'.csv', sep =',', header=False, index=False, na_rep=-1, float_format='%d', columns=FSWLabel)
    TrD.to_csv('../data/folds/'+fileName+woeb+alg+'.discr.train.p'+str(i)+'.PM', sep =' ', header=False, index=False, na_rep=999999, float_format='%d', columns=FS)
    Te.to_csv('../data/folds/'+fileName+woeb+alg+'.test.p'+str(i)+'.csv', sep =',', header=False, index=False, na_rep=-1, columns=FSWLabel)
    TeD.to_csv('../data/folds/'+fileName+woeb+alg+'.discr.test.p'+str(i)+'.csv', sep =',', header=False, index=False, na_rep=-1, float_format='%d', columns=FSWLabel)

# Printing private data
filesTF = ['2019-08', '2019-09', '2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04', '2020-05']
for file in filesTF:
    TF = pd.read_csv('../dataPrivate/'+file+'.Merged'+woeb+'.csv', sep =',', header=None, na_values=-1, index_col=0)
    TF.columns = range(TF.shape[1])
    TF.to_csv('../dataPrivate/'+file+'.Merged'+woeb+alg+'.csv', sep =',', header=False, na_rep=-1, columns=FSWLabel)
    
    TF = pd.read_csv('../dataPrivate/'+file+'.Merged'+woeb+'.discr.csv', sep =',', header=None, na_values=-1, index_col=0)
    TF.columns = range(TF.shape[1])
    TF.to_csv('../dataPrivate/'+file+'.Merged'+woeb+alg+'.discr.csv', sep =',', header=False, na_rep=-1, float_format='%d', columns=FSWLabel)
    if file == '2019-08':
        TF.to_csv('../dataPrivate/'+file+'.Merged'+woeb+alg+'.discr.PM', sep =' ', header=False, index=False, na_rep=999999, float_format='%d', columns=FS)
