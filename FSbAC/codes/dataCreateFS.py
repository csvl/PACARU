#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:47:06 2023

@author: rve
"""
import pandas as pd
import pickle

### DATA CONSIDERING ONLY THE SELECTED FEATURES


fileInfoAttr = 'attrInfoWoEB'
fileName = 'dataConcatWoEB'

infoAttr = pd.read_csv('../data/'+fileInfoAttr+'.csv', sep =',', header=0, index_col=None)
idxlabel = infoAttr.shape[0]


# Loop in the 10-folds
for i in range(10):
    Tr = pd.read_csv('../data/folds/'+fileName+'.train.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    Te = pd.read_csv('../data/folds/'+fileName+'.test.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    TrD = pd.read_csv('../data/folds/'+fileName+'.discr.train.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    TeD = pd.read_csv('../data/folds/'+fileName+'.discr.test.p'+str(i)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
    infile = open('../ResultsPM/rules/'+fileName+'.discr.MG.p'+str(i)+'.FSbAC4', 'rb')
    rules = pickle.load(infile)
    infile.close()
    
    FS = set()
    for rule in rules:
        FS = FS.union(set(rule[0]))
    FS = list(FS)
    FS.sort()
    FSWLabel = FS.copy()
    FSWLabel.append(idxlabel)
    
    infoAttrNew = infoAttr.iloc[FS,:]
    
    infoAttrNew.to_csv('../data/folds/'+fileInfoAttr+'FSbAC.p'+str(i)+'.csv', sep =',', header=True, index=False)
    Tr.to_csv('../data/folds/'+fileName+'FSbAC.train.p'+str(i)+'.csv', sep =',', header=False, index=False, na_rep=-1, columns=FSWLabel)
    TrD.to_csv('../data/folds/'+fileName+'FSbAC.discr.train.p'+str(i)+'.csv', sep =',', header=False, index=False, na_rep=-1, float_format='%d', columns=FSWLabel)
    TrD.to_csv('../data/folds/'+fileName+'FSbAC.discr.train.p'+str(i)+'.PM', sep =' ', header=False, index=False, na_rep=999999, float_format='%d', columns=FS)
    Te.to_csv('../data/folds/'+fileName+'FSbAC.test.p'+str(i)+'.csv', sep =',', header=False, index=False, na_rep=-1, columns=FSWLabel)
    TeD.to_csv('../data/folds/'+fileName+'FSbAC.discr.test.p'+str(i)+'.csv', sep =',', header=False, index=False, na_rep=-1, float_format='%d', columns=FSWLabel)