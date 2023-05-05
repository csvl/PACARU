#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 17:16:27 2023

@author: rve
"""
import pandas as pd
import numpy as np

### REMOVE THE Entry Bytes EB ATTRIBUTES FROM THE DATASETS

filesTF = ['2019-08', '2019-09', '2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04', '2020-05']

infoAttr = pd.read_csv('../data/attrInfo.csv', sep =',', header=0, index_col=0)
D = pd.read_csv('../data/dataConcat.csv', sep =',', header=None, na_values=-1, index_col=None)
Ddiscr = pd.read_csv('../data/dataConcat.discr.csv', sep =',', header=None, na_values=-1, index_col=None)
minsupsF = pd.read_csv('../data/dataConcat.discr.minsups.csv', sep =',', header=0, index_col=None)


infoAttrNew = infoAttr[infoAttr.Group != 'Entry Bytes EB']
index = infoAttrNew.index.to_list()
indexWLabel = index.copy()
indexWLabel.append(119)


#Printing public data
infoAttrNew.to_csv('../data/attrInfoWoEB.csv', sep =',', header=True, index=True)
D.to_csv('../data/dataConcatWoEB.csv', sep =',', header=False, index=False, na_rep=-1, columns=indexWLabel)
Ddiscr.to_csv('../data/dataConcatWoEB.discr.csv', sep =',', header=False, index=False, na_rep=-1, float_format='%d', columns=indexWLabel)
Ddiscr.to_csv('../data/dataConcatWoEB.discr.PM', sep =' ', header=False, index=False, na_rep=999999, float_format='%d', columns=index)

# Loop in the 10-folds
for i in range(10):
    Tr = pd.read_csv('../data/folds/dataConcat.train.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    Te = pd.read_csv('../data/folds/dataConcat.test.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    TrD = pd.read_csv('../data/folds/dataConcat.discr.train.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    TeD = pd.read_csv('../data/folds/dataConcat.discr.test.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    
    nlabel = np.unique(Tr[119], return_counts=True)
    minsups = np.floor(minsupsF['minsupWoEB'] * nlabel[1])
    
    Tr.to_csv('../data/folds/dataConcatWoEB.train.p'+str(i)+'.csv', sep =',', header=False, index=False, na_rep=-1, columns=indexWLabel)
    TrD.to_csv('../data/folds/dataConcatWoEB.discr.train.p'+str(i)+'.csv', sep =',', header=False, index=False, na_rep=-1, float_format='%d', columns=indexWLabel)
    TrD.to_csv('../data/folds/dataConcatWoEB.discr.train.p'+str(i)+'.PM', sep =' ', header=False, index=False, na_rep=999999, float_format='%d', columns=index)
    Te.to_csv('../data/folds/dataConcatWoEB.test.p'+str(i)+'.csv', sep =',', header=False, index=False, na_rep=-1, columns=indexWLabel)
    TeD.to_csv('../data/folds/dataConcatWoEB.discr.test.p'+str(i)+'.csv', sep =',', header=False, index=False, na_rep=-1, float_format='%d', columns=indexWLabel)
    minsups.to_csv('../data/folds/dataConcatWoEB.discr.minsups.p'+str(i), sep =' ', header=False, index=True, float_format='%d')

#Printing private data    
for file in filesTF:
    TF = pd.read_csv('../dataPrivate/'+file+'.Merged.csv', sep =',', header=None, na_values=-1, index_col=0)
    TF.columns = range(TF.shape[1])
    TF.to_csv('../dataPrivate/'+file+'.MergedWoEB.csv', sep =',', header=False, na_rep=-1, columns=indexWLabel)
    
    TF = pd.read_csv('../dataPrivate/'+file+'.Merged.discr.csv', sep =',', header=None, na_values=-1, index_col=0)
    TF.columns = range(TF.shape[1])
    TF.to_csv('../dataPrivate/'+file+'.MergedWoEB.discr.csv', sep =',', header=False, na_rep=-1, float_format='%d', columns=indexWLabel)
    if file == '2019-08':
        TF.to_csv('../dataPrivate/'+file+'.MergedWoEB.discr.PM', sep =' ', header=False, index=False, na_rep=999999, float_format='%d', columns=index)
