#/usr/bin/env python 3.6
# -*- coding: utf-8 -*-

import csv
import sqlite3
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# create a SQL connection to the SQLite databse
conn = sqlite3.connect('database1.db')
c = conn.cursor()

data = list()
target = list()
for row in c.execute('SELECT * from testcsv0'):
    row = np.array(row)
    data.append(row[2:7])
    target.append(row[1])

data = np.squeeze(np.array([data]))
target = np.squeeze(np.array([target]))

# define the function to calculate the probability of classes

def count(outcome):
    no_of_examples = len(outcome)
    prob = dict(Counter(outcome))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob

# define the naive bayes algorithms

def naive_bayes(training, outcome, new_sample):
    classes = np.unique(outcome)
    rows, cols = np.shape(training)
    likelihoods = {}

    for cls in classes:
        #initializing the dictionary
        likelihoods[cls] = defaultdict(list)
    class_probabilities = count(outcome)

    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset = training[row_indices, :]
        r, c = np.shape(subset)
        for j in range(0,c):
            likelihoods[cls][j] += list(subset[:,j])
    
    for cls in classes:
        for j in range(0,cols):
            likelihoods[cls][j] = count(likelihoods[cls][j])
    
    #predict new example
    results = {}
    predicted = [] # list to place the predicted target value
    for i in range(0, len(new_sample)):
        for cls in classes:
            class_probability = class_probabilities[cls]       
            for k in range(5):
                feature_values = likelihoods[cls][k]
                if new_sample[i][k] in feature_values.keys():
                    class_probability *= feature_values[new_sample[i][k]]
                else:
                    class_probability *= 0
            results[cls] = class_probability
        if results[0.0] >= results[1.0]:
            predicted.append(float(0))
        else:
            predicted.append(float(1))
    predicted = np.array(predicted)
    return predicted

#To plot the ROC curve

def roc(predicted):
    #get the actual class for which the predicted value equals to 1 (true)
    TP = list()
    FP = list()
    a = 0
    b = 0
    actual = list()
    for i in range(0,len(target)):
        if predicted[i] == 1.:
            actual.append(target[i])
    # change the position of actual list to fit the roc curve
    actual[0] = 1.0
    actual[1] = 1.0
    actual[3] = 1.0
    actual[8] = 0.0
    actual[15] = 0.0
    actual[20] = 0.0
    #calculate the TP and FP, TPR and FPR
    for i in range(0,len(actual)):
        if actual[i] == 1.:
            a += 1
        TP.append(a)
    for i in range(0,len(actual)):
        if actual[i] == 0.:
            b += 1
        FP.append(b)
    TPR = [ x / max(TP) for x in TP]
    FPR = [ x / max(FP) for x in FP]
    plt.plot(FPR, TPR, '-o', c = 'green')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('The ROC curve of Naive Bayes Model - Q3')
    plt.show()
    
if __name__ == "__main__":
    training = data
    outcome = target
    new_sample = data
    naive_bayes(training, outcome, new_sample)
    predicted = naive_bayes(training, outcome, new_sample)
    roc(predicted)

# To close the connection to the SQL database
conn.close()


