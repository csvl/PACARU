#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:56:37 2022

@author: rve
"""
import pandas as pd


print("Public data")

d1 = pd.read_csv('../data/not-packed_output.csv', sep =',', header=None, na_values=-1, index_col=0)
d1.columns = range(d1.shape[1])
d2 = pd.read_csv('../data/packed_feature_output.csv', sep =',', header=None, na_values=-1, index_col=0)
d2.columns = range(d2.shape[1])

d3 = pd.read_csv('../data/kkrunchy.csv', sep =',', header=None, na_values=-1, index_col=None)
d3[119] = d3.shape[0] * ['kkrunchy']

d4 = pd.read_csv('../data/armadillo.csv', sep =',', header=None, na_values=-1, index_col=None)
d4[119] = d4.shape[0] * ['armadillo']

d5 = pd.read_csv('../data/enigma.csv', sep =',', header=None, na_values=-1, index_col=None)
d5[119] = d5.shape[0] * ['enigma']

d6 = pd.read_csv('../data/VMProtect.csv', sep =',', header=None, na_values=-1, index_col=None)
d6[119] = d6.shape[0] * ['vmprotect']

df = pd.concat([d1, d2, d3, d4, d5, d6], ignore_index=True)
df.index = range(df.shape[0])
df.to_csv('../data/dataConcat.csv', sep =',', header=False, index=False, na_rep=-1)
