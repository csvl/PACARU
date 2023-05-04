#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:12:57 2022

A bicluster is a submatrix of a data matrix such that the rows in the submatrix
exhibit a consistent pattern across its columns (and/or vice-versa).

A formal concept is a maximal bicluster of 1’s in a binary data matrix.

A closed itemset is the intent (set of columns) of a formal concept.


@author: rve
"""

import numpy as np
import pandas as pd


# Reads a Patter Mining solution
def readPMOutput(filename):
    bics = []
    with open(filename) as bicfile:
        for line in bicfile:
            exec(line)
    return bics

# Get information about the biclusters (patterns)
def getInfoPatterns(bics, labels, infoAttr):
    n = labels.shape[0]
    counts = labels.value_counts()
    res = []
    bicsRowsClass = []
    costPerGroup = getCostPerGroup(infoAttr)
    for bic in bics:
        rows = np.array(bic[0])
        cols = np.array(bic[1])
        sizeRows = rows.shape[0]
        sizeCols = cols.shape[0]
        (classe, rowsClass) = getClassBic(rows, labels)
        sizeRowsClass = rowsClass.shape[0]
        rsupX = sizeRows / n
        rsupY = counts[classe] / n
        rsupXY = sizeRowsClass / n;
        conf = sizeRowsClass / sizeRows
        comp = sizeRowsClass / counts[classe]
        lift = conf / rsupY
        leverage = rsupXY - rsupX * rsupY
        bicsRowsClass.append(rowsClass)
        costTotal = infoAttr['Time Cost Median'][cols].sum()
        costGroups = getCostPerGroupCols(infoAttr, costPerGroup, cols)
        res.append([sizeRows, sizeCols, classe, sizeRowsClass, rsupX, rsupY, rsupXY, conf, comp, lift, leverage, costTotal, costGroups])
    df = pd.DataFrame(res, columns=['sizeRows', 'sizeCols', 'ClassLabel', 'sizeRowsClass', 'rsupX', 'rsupY', 'rsupXY', 'Confidence', 'Completeness', 'Lift', 'Leverage', 'costTotal', 'costGroups'])
    return (df, bicsRowsClass)

def getCostPerGroup(infoAttr):
    Group = infoAttr['Group']
    Cost = infoAttr['Time Cost Median']
    costPerGroup = {}
    for i in range(0, Group.shape[0]): 
        if not Group[i] in costPerGroup:
            costPerGroup[Group[i]] = Cost[i]
    return costPerGroup

def getCostPerGroupCols(infoAttr, costPerGroup, cols):
    grupos = infoAttr['Group'][cols].unique()
    custo = 0
    for grupo in grupos:
        custo = custo + costPerGroup[grupo]
    return custo

# Get the main class of a bicluster and the rows that belong to that class label
def getClassBic(rows, labels):
    labelsOfRows = labels[rows]
    counts = labelsOfRows.value_counts()
    classe = counts.idxmax()
    rowsClass = rows[labelsOfRows == classe]
    return (classe, rowsClass)


# Liu, B., Hsu, W., & Ma, Y. (1998, August). Integrating classification and association rule mining. In Kdd (Vol. 98, pp. 80-86).
def CBA_selection_M1(bics, infoBics, bicsRowsClass, Ytrain, ranking):
    newSummary = infoBics.copy()
    newSummary = newSummary.sort_values(ranking[0], ascending = ranking[1])
    
    notCoveredRows = np.ones(shape=[Ytrain.shape[0]], dtype=int)
    idxSelectedBics = []
    sumFP = 0
    # FPs = []
    # errosElse = []
    errosTotal = []
    elses = []
    for index in newSummary.index:
        # if the rule correctly classifies at least one instance
        if sum(notCoveredRows[bicsRowsClass[index]]) >= 1:
            FP = sum(notCoveredRows[bics[index][0]]) - sum(notCoveredRows[bicsRowsClass[index]])
            notCoveredRows[bics[index][0]] = 0 # "delete" all the cases covered by the select rule
            labelsToCover = Ytrain[notCoveredRows == 1]
            if labelsToCover.shape[0] == 0:
                # This last rule has conf < 100% and it is covering all remaing training instances.
                # So, it does not decrease the ToTal Error of the classifier.
                break;            
            vcLabelsToCover = labelsToCover.value_counts()
            sumFP = sumFP + FP # sum of false positive in the current rule-list-classifier
            erroElse = notCoveredRows.sum() - vcLabelsToCover[0]
            erroTotal = sumFP + erroElse
            idxSelectedBics.append(index) #insert the rule at the end of the rule-list-classifier
            # FPs.append(FP)
            # errosElse.append(erroElse)
            errosTotal.append(erroTotal)
            elses.append(vcLabelsToCover.index[0])
        if notCoveredRows.sum() == 0 or erroTotal == 0 or vcLabelsToCover.shape[0] <= 1:
            break
    
    # Prune phase:
    idx = np.argmin(errosTotal)
    default_label = elses[idx]
    idxSelectedBicsFinal = idxSelectedBics[:idx+1] 
    return idxSelectedBicsFinal, default_label#, idxSelectedBics, FPs, errosElse, errosTotal, elses


# Chen, F., Wang, Y., Li, M., Wu, H., & Tian, J. (2014). Principal association mining: an efficient classification approach. Knowledge-Based Systems, 67, 16-25.
# ORIGINAL PAM: all the conflicting rules (rules with the same condition but poiting to different class labels) that satisfy the user-defined minimum confidence are retained as candidate rules.
# THIS IMPLEMENTATION works as original PAM only if minConf > 50% because then we have only one optimal rule for each condition.
def PAM_selection(bics, infoBics, bicsRowsClass, n_samples, Lambda=0.8, wGroup=True):
    ranking = [["Principality", "Confidence", "rsupXY", "sizeCols"], (False, False, False, False)]
    newSummary = infoBics.copy()
    
    # Pruning by group
    if wGroup:
        newSummary = PAM_grouping(bics, newSummary)
    
    # Calculate the Principality for each rule in newSummary
    newSummary['Principality'] = Lambda * newSummary.Confidence + (1 - Lambda) * newSummary.Completeness
        
    #Rule Rank
    newSummary = newSummary.sort_values(ranking[0], ascending = ranking[1])
    
    covered = set()
    idxSelectedBics = []
    while newSummary.shape[0] > 0:
        ibic = newSummary.index[0] # index of the selected bicluster
        idxSelectedBics.append(ibic)
        covered = covered.union(set(bics[ibic][0])) # paper2
        
        # Update Principality for each rule in newSummary
        for index in newSummary.index:
            new_coverage = set(bicsRowsClass[index]) - covered
            newSummary.at[index, 'sizeRowsClass'] = len(new_coverage)
            newSummary.at[index, 'Completeness'] = (newSummary.at[index, 'sizeRowsClass'] / n_samples) / newSummary.at[index, 'rsupY']
            newSummary.at[index, 'Principality '] = Lambda * newSummary.at[index, 'Confidence'] + (1 - Lambda) * newSummary.at[index, 'Completeness']
        
        # Delete all rules that do not cover any object
        # (So, rules that do not classify none of the remaining objects correctly)
        newSummary = newSummary[newSummary.Completeness > 0]
        
        #Rule Rank
        newSummary = newSummary.sort_values(ranking[0], ascending = ranking[1])
    
    # Return the selected biclusters (rules)
    return idxSelectedBics

# All rules are categorized into groups according to their class labels. Then the rules with lower confidence subsumed by more specific rules with higher confidence will be pruned.
def PAM_grouping(bics, newSummary):
    labelsvc = newSummary.ClassLabel.value_counts()
    for label in labelsvc.index:
        subset  = newSummary[newSummary.ClassLabel == label]
        subset = subset.sort_values(["Confidence", "Completeness", "sizeCols"], ascending = (False, False, False))
        if subset.Confidence.iloc[0] == 1 and subset.Completeness.iloc[0] == 1:
            # print(label)
            selecao = subset.index[1:].tolist()
            newSummary.drop(selecao, inplace=True)
            continue
        # print(label)
        deletar = np.zeros(shape=[subset.shape[0],])
        for i1 in range(0, subset.shape[0]-1):
            bici1 = bics[subset.index[i1]]
            Ai1 = set(bici1[0])
            Bi1 = set(bici1[1])
            for i2 in range(i1+1, subset.shape[0]):
                if deletar[i2] == 0 and subset.Confidence.iloc[i1] > subset.Confidence.iloc[i2]:
                    bici2 = bics[subset.index[i2]]
                    Ai2 = set(bici2[0])
                    Bi2 = set(bici2[1])
                    if Bi2.issubset(Bi1) and Ai1.issubset(Ai2):
                        deletar[i2] = 1
        selecao = subset.index[np.where(deletar == 1)].tolist()
        newSummary.drop(selecao, inplace=True)
    return newSummary


#Dam, K. H. T., Given-Wilson, T., Legay, A., & Veroneze, R. (2022). Packer classification based on association rule mining. Applied Soft Computing, 127, 109373.
def getBestRulePerLabel (bics, infoBics, bicsRowsClass, n_samples, default_label, Lambda=0.8):
    ranking = [["Principality", "Confidence", "rsupXY", "sizeCols"], (False, False, False, False)]
    newSummary = infoBics.copy()
    newSummary = newSummary[newSummary.ClassLabel != default_label]
    
    # Calculate the Principality for each rule in newSummary
    newSummary['Principality'] = Lambda * newSummary.Confidence + (1 - Lambda) * newSummary.Completeness
        
    #Rule Rank
    newSummary = newSummary.sort_values(ranking[0], ascending = ranking[1])
    
    idxSelectedBics = []
    while newSummary.shape[0] > 0:
        label = newSummary.ClassLabel.iat[0]
        ibic = newSummary.index[0] # index of the selected bicluster
        idxSelectedBics.append(ibic)
        newSummary = newSummary[newSummary.ClassLabel != label] # Delete all rules from the same class label
    
    # Return the selected biclusters (rules)
    return idxSelectedBics


def FSbAC(bics, infoBics, bicsRowsClass, n_samples, ranking, infoAttr, minCorrectlyClassified=1, default_label=''):
    newSummary = infoBics.copy()
    if default_label !=  '':
        newSummary = newSummary[newSummary.ClassLabel != default_label]
    newSummary = newSummary.sort_values(ranking[0], ascending = ranking[1])
    
    costPerGroup = getCostPerGroup(infoAttr)
    costIndividual = infoAttr['Time Cost Median'].copy()
    countAttr = np.ones(costIndividual.shape)
    
    coveredRows = np.zeros(shape=n_samples, dtype=int)
    idxSelectedBics = []
    while newSummary.shape[0] > 0:
        index = newSummary.index[0] # index of the current top ranked rule
        if sum(coveredRows[bicsRowsClass[index]] < 1) >= minCorrectlyClassified:
            coveredRows[bics[index][0]] = 1
            idxSelectedBics.append(index)
            
            if (coveredRows > 0).all(): break
            
            #Update costs
            costIndividual[bics[index][1]] = 0 # zero the cost of the features in the selected rule
            countAttr[bics[index][1]] = 0
            for grupo in infoAttr['Group'][bics[index][1]].unique():
                costPerGroup[grupo] = 0
            
            # Update some metrics for each remaining rule in newSummary
            sizeRowsClass = []
            rsupXYs = []
            Completeness = []
            costTotal = []
            costGroups = []
            sizeCols = []
            for index2 in newSummary.index:
                new_bicsRowsClass = set(bicsRowsClass[index2]) - set(np.nonzero(coveredRows)[0])
                supXY = len(new_bicsRowsClass)
                rsupXY = supXY / n_samples
                completeness = rsupXY / newSummary.at[index2, 'rsupY']
                sizeRowsClass.append(supXY)
                rsupXYs.append(rsupXY)
                Completeness.append(completeness)
                costTotal.append(costIndividual[bics[index2][1]].sum())
                costGroups.append(getCostPerGroupCols(infoAttr, costPerGroup, bics[index2][1]))
                sizeCols.append(countAttr[bics[index2][1]].sum())
            newSummary.loc[:, ['sizeRowsClass', 'rsupXY', 'Completeness', 'costTotal', 'costGroups', 'sizeCols']] = list(zip(sizeRowsClass, rsupXYs, Completeness, costTotal, costGroups, sizeCols))
            
            newSummary = newSummary[newSummary.Completeness != 0]
            newSummary = newSummary.sort_values(ranking[0], ascending = ranking[1])
        else:
            newSummary.drop(index, inplace=True)
    
    # Return the selected biclusters (rules)
    return idxSelectedBics


# Build CARs with the bics in idxSelectedBics
def build_CARs (bics, idxSelectedBics, summary, D_train):
    rules = list()    
    for idx in idxSelectedBics:
        bic = bics[idx]
        dbic = D_train.iloc[bic[0],bic[1]]
        rules.append((bic[1], dbic.min(), dbic.max(), summary.ClassLabel[idx]))
    return rules


# Get the predicted labels for test data instances
def RuleListPredictions (rules, D_test, default_label):
    yhat = [default_label for i in range(D_test.shape[0])]
    
    i = 0
    for i in range(D_test.shape[0]):
        for rule in rules:
            vlrs = D_test[i][rule[0]]
            if (vlrs >= rule[1]).all() and (vlrs <= rule[2]).all():
                yhat[i] = rule[3]
                break
        i = i + 1
    return yhat


# Get information about the attributes in rules
def getInfoAttrRules(rules, infoAttr):
    colsUnicas = set()
    nComp = 0
    for rule in rules:
        colsUnicas = colsUnicas.union(rule[0])
        nComp = nComp + len(rule[0])
    colsUnicas = list(colsUnicas)
    custoTotal = infoAttr['Time Cost Median'][colsUnicas]
    costPerGroup = getCostPerGroup(infoAttr)
    custoGrupo = getCostPerGroupCols(infoAttr, costPerGroup, colsUnicas)
    return colsUnicas, nComp, custoTotal.sum(), custoGrupo
