#! /usr/bin/env python
# -*- coding: utf-8 -*-


def firstRank(predicted, gold):
    """returns the the rank of the first correct predicted keyphrase"""
    firstRank = 0
    for i in range(0, len(predicted)):
        if predicted[i] in gold:
            firstRank = i
            break

    return firstRank


def Rprecision(predicted, gold, k):

    hits = set(predicted).intersection(set(gold))
    Rpr = 0.0
    if len(hits)>0 and len(predicted)>0:
        Rpr =  len(hits)*1.0/k

    return Rpr

def PRF(predicted, gold, k):

    predicted = predicted[:k]

    hits = set(predicted).intersection(set(gold))
    P, R, F1 = 0.0, 0.0, 0.0

    if len(hits)>0 and len(predicted)>0:
        P =  len(hits)/len(predicted)
        R = len(hits)/len(gold)
        F1 = 2*P*R/(P+R)

    return {'precision':P,'recall': R,'f1-score': F1}

def PRF_range(predicted, gold, k):

    P = []
    R = []
    F1 = []

    for i in range(0,k):
        predict = predicted[:i+1]
        
        hits = set(predict).intersection(set(gold))
        pr = 0.0
        re = 0.0
        f1 = 0.0
        if len(hits)>0 and len(predict)>0:
            pr =  len(hits)*1.0/len(predict)
            re = len(hits)*1.0/len(gold)
        if pr+re > 0:
            f1 = 2*pr*re/(pr+re)
        
        P.append(pr)
        R.append(re)
        F1.append(f1)

    return P,R,F1

def Bpref (pred, gold):
    incorrect = 0
    correct = 0
    bpref = 0

    for kp in pred:
        if kp in gold:
            bpref += (1.0 - (incorrect*1.0/len(pred)))
            correct += 1
        else:
            incorrect +=1

    if correct >0:
        bpref = bpref*1.0/correct
    else:
        bpref = 0.0

    return bpref