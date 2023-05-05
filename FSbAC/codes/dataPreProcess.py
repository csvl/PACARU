#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:37:50 2022

@author: rve
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder


def replaceLabels(Y, diferenca):
    for label in diferenca:
        Y[Y == label] = 'other packer'
    return Y


files = ['2019-08', '2019-09', '2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04', '2020-05']
labelsToKeep = {'not packed', 'upx', 'vmprotect', 'aspack', 'telock', 'kkrunchy', 'pecompact', 'mpress', 'enigma', 'armadillo', 'neolite', 'nspack', 'pacman'}


print("Loading data")
dAttrInfo = pd.read_csv('../data/attrInfo.csv', sep =',', header=0, index_col=0)
dPublic = pd.read_csv('../data/dataConcat.csv', sep =',', header=None, na_values=-1, index_col=None)

DTFs = [[]] * len(files)
for i, file in enumerate(files):
    df = pd.read_csv('~/Documents/github/BiClusterDetectPack/DataSets/'+file+'.Merged.csv', sep =',', header=None, na_values=-1, index_col=0)
    df.columns = range(df.shape[1])
    
    diferenca = list(set(df.iloc[:,-1]) - labelsToKeep)
    df.iloc[:,-1] = replaceLabels(df.iloc[:,-1], diferenca)
    df.to_csv('../dataPrivate/'+file+'.Merged.csv', sep =',', header=False, na_rep=-1)
    
    DTFs[i] = df.copy()


dPublicD = dPublic.copy()
for ncol in range(dAttrInfo.shape[0]):
    print("Column =", ncol, " Partition =", dAttrInfo.Partition[ncol], " name=", dAttrInfo.Feature[ncol])
    tf_digitize = True
    if dAttrInfo.Partition[ncol] == 'B' or dAttrInfo.Partition[ncol] == 'N': # binary or categorical column
        aux = dPublic[ncol].value_counts()
        if len(aux) == 1: #all values are the same
            dPublicD.iloc[:, ncol] = np.NAN
    else: # columns to be partitioned
        nbins = int(dAttrInfo.nbins[ncol])
        print("n_bins", nbins)
        ile = -1
        
        idxPublic = np.where(~dPublic[ncol].isnull())[0]
        if dAttrInfo.Partition[ncol][0] == 'Q':
            kbin_dis = KBinsDiscretizer(n_bins=nbins, encode='ordinal', strategy='quantile')
            valoresn = dPublic.iloc[idxPublic, ncol]
            if dAttrInfo.Partition[ncol] == 'Q':
                print("Quantile-based partition")
                tf_digitize = False
                vsPubic = kbin_dis.fit_transform(valoresn.to_numpy().reshape(-1, 1))
                bins = kbin_dis.bin_edges_[0]
            elif dAttrInfo.Partition[ncol] == 'Q1':
                print("Keep the value 0 and partition the remainder")
                valoresn = valoresn[valoresn>0]
                kbin_dis.fit(valoresn.to_numpy().reshape(-1, 1))
                bins = kbin_dis.bin_edges_[0]
                bins = np.insert(bins, 0, 0)
                bins[-1] = np.Inf # This is done for the method np.digitize works propoerly
                vsPubic = np.digitize(dPublic.iloc[idxPublic, ncol], bins)
            elif dAttrInfo.Partition[ncol] == 'Q2':
                print("Keep the values 0 and 1 and partition the remainder")
                valoresn = valoresn[valoresn>1]
                kbin_dis.fit(valoresn.to_numpy().reshape(-1, 1))
                bins = kbin_dis.bin_edges_[0]
                bins = np.insert(bins, 0, 0)
                bins = np.insert(bins, 1, 1)
                bins[-1] = np.Inf # This is done for the method np.digitize works propoerly
                vsPubic = np.digitize(dPublic.iloc[idxPublic, ncol], bins)
            elif dAttrInfo.Partition[ncol] == 'Q3':
                print("Keep the values 0, 1 and 2, and partition the remainder")
                valoresn = valoresn[valoresn>2]
                kbin_dis.fit(valoresn.to_numpy().reshape(-1, 1))
                bins = kbin_dis.bin_edges_[0]
                bins = np.insert(bins, 0, 0)
                bins = np.insert(bins, 1, 1)
                bins = np.insert(bins, 2, 2)
                bins[-1] = np.Inf # This is done for the method np.digitize works propoerly
                vsPubic = np.digitize(dPublic.iloc[idxPublic, ncol], bins)
            elif dAttrInfo.Partition[ncol] == 'Q4':
                print("First bin: <2. Secon bin: =2. Partition the remainder")
                valoresn = valoresn[valoresn>2]
                kbin_dis.fit(valoresn.to_numpy().reshape(-1, 1))
                bins = kbin_dis.bin_edges_[0]
                bins = np.insert(bins, 0, -np.Inf)
                bins = np.insert(bins, 1, 2)
                bins[-1] = np.Inf # This is done for the method np.digitize works propoerly
                vsPubic = np.digitize(dPublic.iloc[idxPublic, ncol], bins)
            elif dAttrInfo.Partition[ncol] == 'Q5':
                print("Keep the minimum and maximum values (zero and one) and partition the remainder")
                ile = -2
                minimo = valoresn.min()
                maximo = valoresn.max()
                valoresn = valoresn[valoresn>minimo]
                valoresn = valoresn[valoresn<maximo]
                kbin_dis.fit(valoresn.to_numpy().reshape(-1, 1))
                bins = kbin_dis.bin_edges_[0]
                bins = np.insert(bins, 0, minimo)
                bins[-1] = maximo
                np.append(bins, np.Inf) # This is done for the method np.digitize works propoerly
                vsPubic = np.digitize(dPublic.iloc[idxPublic, ncol], bins)
            print("Bin_edges", kbin_dis.bin_edges_)
        elif dAttrInfo.Partition[ncol] == 'F':
            print("Pre-defined bins")
            bins = np.arange(0,nbins,1)
            bins = np.append(bins, np.Inf)
            print("Bin_edges", bins)
            vsPubic = np.digitize(dPublic.iloc[idxPublic, ncol], bins)
            
        dPublicD.iloc[idxPublic, ncol] = vsPubic.reshape(-1)
        print("Public", dPublicD.iloc[idxPublic, ncol].value_counts().to_string())
        for i in range(len(files)):
            idxPrivate = np.where(~DTFs[i][ncol].isnull())[0]
            vsPrivate = np.digitize(DTFs[i].iloc[idxPrivate, ncol], bins)
            vsPrivate = kbin_dis.transform((DTFs[i].iloc[idxPrivate, ncol]).to_numpy().reshape(-1, 1))
            DTFs[i].iloc[idxPrivate, ncol] = vsPrivate.reshape(-1)
            

# Printing the private data
for i, file in enumerate(files):
    DTFs[i].to_csv('../dataPrivate/'+file+'.Merged.discr.csv', sep =',', header=False, na_rep=-1, float_format='%d')
DTFs[0].to_csv('../dataPrivate/'+files[0]+'.Merged.discr.PM', sep =' ', header=False, index=False, na_rep=999999, float_format='%d', columns=range(DTFs[0].shape[1]-1))
le = LabelEncoder()
labels = le.fit_transform(DTFs[0].iloc[:,-1])
np.savetxt('../dataPrivate/'+files[0]+'.Merged.discr.labels', labels, fmt='%d')

# Printing the public data
le = LabelEncoder()
labels = le.fit_transform(dPublicD.iloc[:,-1])
np.savetxt('../data/dataConcat.labels', labels, fmt='%d')
dPublicD.to_csv('../data/dataConcat.discr.csv', sep =',', header=False, index=False, na_rep=-1, float_format='%d')
dPublicD.to_csv('../data/dataConcat.discr.PM', sep =' ', header=False, index=False, na_rep=999999, float_format='%d', columns=range(dPublicD.shape[1]-1))
