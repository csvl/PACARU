#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:11:49 2022

@author: rve
"""

import time
import pandas as pd
import numpy as np
import pickle
import pattern_mining as pm
from sys import argv


# fileInfoAttr = argv[1]
# fileName = argv[2]
# tipo = argv[3]
# nfold = int(argv[4])
fileInfoAttr = 'folds/attrInfoWoEBLupxFS.p'
fileName = 'dataConcatWoEBLupxFS.discr'
tipo = 'MG' # CI or MG or CI.ms10...
nfold = 9
if 'folds' in fileInfoAttr:
    fileInfoAttr += str(nfold)
print("fileInfoAttr:", fileInfoAttr)
print("fileName:", fileName)
print("CI or MG?:", tipo)
print("nfold:", nfold)


pathData = '../data/folds/'
pathPM = '../ResultsPM/'
default_label = 0 #'not packed'
#algs = ['PAM', 'PAM-WoG', 'BR'] if tipo=='CI' else ['CBA', 'CBAcomp', 'FSbAC']
algs = ['PAM', 'PAM-WoG', 'BR'] if 'CI' in tipo else ['CBA', 'CBAcomp']


runtimes = []


print('Loading data and biclusters')
infoAttr = pd.read_csv('../data/'+fileInfoAttr+'.csv', sep =',', header=0, index_col=None)
Dtrain = pd.read_csv(pathData+fileName+'.train.p'+str(nfold)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
Dtest = pd.read_csv(pathData+fileName+'.test.p'+str(nfold)+'.csv', sep =',', header=None, na_values=-1, index_col=None)
bics = pm.readPMOutput(pathPM+'bics/'+fileName+'.'+tipo+'.p'+str(nfold)+'.py')


print('Getting Information about the Pattern Mining solution')
inicio = time.time()
(infoBics, bicsRowsClass) = pm.getInfoPatterns(bics, Dtrain.iloc[:,-1], infoAttr)
segsInfoBics = time.time() - inicio; print('Finished in', segsInfoBics, 'seconds.')
infoBics.to_csv(pathPM+'infoBics1/'+fileName+'.'+tipo+'.p'+str(nfold)+'.csv', sep=',', header=True, index=True)
with open(pathPM+'infoBics1/'+fileName+'.'+tipo+'.p'+str(nfold)+'.bicsRowsClass', 'wb') as filehandle:
    pickle.dump(bicsRowsClass, filehandle) # store the data as binary data stream


for alg in algs:
    print(alg)
    msg = '.' + alg
    runtimes.append(segsInfoBics)
    print('Selecting rules for the classifier')
    inicio = time.time()
    if alg == 'CBA':
        ranking = [["Confidence", "rsupXY", "sizeCols"],(False, False, True)]
        idxSelectedBics, default_label = pm.CBA_selection_M1(bics, infoBics, bicsRowsClass, Dtrain.iloc[:,-1], ranking)
    elif alg == 'CBAcomp':
        ranking = [["Confidence", "Completeness", "rsupXY", "sizeCols"],(False, False, False, True)]
        idxSelectedBics, default_label = pm.CBA_selection_M1(bics, infoBics, bicsRowsClass, Dtrain.iloc[:,-1], ranking)
    elif alg == 'PAM':
        idxSelectedBics = pm.PAM_selection(bics, infoBics, bicsRowsClass, Dtrain.shape[0], Lambda=0.8, wGroup=True)
    elif alg == 'PAM-WoG':
        idxSelectedBics = pm.PAM_selection(bics, infoBics, bicsRowsClass, Dtrain.shape[0], Lambda=0.8, wGroup=False)
    elif alg == 'BR':
        idxSelectedBics = pm.getBestRulePerLabel(bics, infoBics, bicsRowsClass, Dtrain.shape[0], default_label, Lambda=0.8)
    elif alg == 'FSbAC':
        ranking = [["Confidence", "Completeness", "rsupXY", "costTotal", "sizeCols"],(False, False, False, True, True)]
        idxSelectedBics = pm.FSbAC(bics, infoBics, bicsRowsClass, Dtrain.shape[0], ranking, infoAttr, minCorrectlyClassified=1, default_label=default_label)
    else:
        print("Unknown classifier")
        exit()
    segs = time.time() - inicio; runtimes.append(segs); print('Finished in', segs, 'seconds.')
    classifier = infoBics.loc[idxSelectedBics, :].copy()
    classifier.to_csv(pathPM+'infoBics2/'+fileName+'.'+tipo+'.p'+str(nfold)+msg+'.csv', sep=',', header=True, index=True)
    
    print('Building rules')
    inicio = time.time()
    rules = pm.build_CARs(bics, idxSelectedBics, infoBics, Dtrain)
    segs = time.time() - inicio; runtimes.append(segs); print('Finished in', segs, 'seconds.')
    with open(pathPM+'rules/'+fileName+'.'+tipo+'.p'+str(nfold)+msg, 'wb') as filehandle:
        pickle.dump(rules, filehandle) # store the data as binary data stream
    
    print('Classifying new instances')
    Ytest = Dtest.iloc[:,-1]
    inicio = time.time()
    Ypred = pm.RuleListPredictions(rules, Dtest.to_numpy(), default_label)
    segs = time.time() - inicio; runtimes.append(segs); print('Finished in', segs, 'seconds.')
    predictions = pd.DataFrame(data=list(zip(Ytest, Ypred)), columns=["True", "Predicted"])
    predictions.to_csv(pathPM+'predictions/'+fileName+'.'+tipo+'.p'+str(nfold)+msg+'.csv', sep=',', header=True, index=False)

pd.DataFrame(data=np.reshape(runtimes,newshape=[len(algs),4]), index=algs).to_csv(pathPM+'runtimes/'+fileName+'.'+tipo+'.p'+str(nfold)+'.csv', sep=',', header=False, index=True)