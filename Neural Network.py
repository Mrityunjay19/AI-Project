import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# hyperparameters
def sigma(x):
    return 1/(1 + np.exp(-x))

def sigma_(x):
    return sigma(x)*(1 - sigma(x))

def cost(y, y_):
    return -1 * (y*np.log(y_) + (1-y)*np.log(1-y_))

structure = np.array([6, 4, 1])
size = len(structure)
alpha = 0.01
epochs = 150

# parameters
params = {}
derivatives = {}

def init():
    for layer in range(1, size):
        params["w" + str(layer)] = np.random.randn(structure[layer], structure[layer - 1]) * 0.01
        params["a" + str(layer)] = np.ones((structure[layer], 1))
        params["b" + str(layer)] = np.ones((structure[layer], 1))
        params["z" + str(layer)] = np.ones((structure[layer], 1))
    
    params["a0"] = np.ones((structure[0], 1))

def feedForward(X):
    params["a0"] = X    
    for layer in range(1, size):
        params["z" + str(layer)] = np.add(np.dot(params["w" + str(layer)], params["a" + str(layer - 1)]), params["b" + str(layer)])
        params["a" + str(layer)] = sigma(params["z" + str(layer)])
    
def backPropagate(y):
    derivatives["z" + str(size - 1)] = params["a" + str(size - 1)] - y
    derivatives["w" + str(size - 1)] = np.dot(derivatives["z" + str(size - 1)], np.transpose(params["a" + str(size - 2)]))
    derivatives["b" + str(size - 1)] = derivatives["z" + str(size - 1)]

    for layer in range(size - 2, 0, -1):
        derivatives["z" + str(layer)] = np.dot(np.transpose(params["w" + str(layer + 1)]), derivatives["z" + str(layer + 1)]) * sigma_(params["z" + str(layer)])
        derivatives["w" + str(layer)] = np.dot(derivatives["z" + str(layer)], np.transpose(params["a" + str(layer - 1)]))
        derivatives["b" + str(layer)] = derivatives["z" + str(layer)]


def gradientDescent():
    for layer in range(1, size):
        params["w" + str(layer)] -= alpha * derivatives["w" + str(layer)]
        params["b" + str(layer)] -= alpha * derivatives["b" + str(layer)]

def predict(X):
    feedForward(X)
    return params["a" + str(size - 1)]

def train(X, Y, epochs = 200):
    for epoch in range(epochs):
        for i in range(X.shape[0]):
            x = X[i].reshape(X[i].size, 1)
            y = Y[i]
            feedForward(x)
            backPropagate(y)
            gradientDescent()
        
            y_pred = predict(x)
            if(y_pred > 0.5):
                y_pred = 1
            else:
                y_pred = 0
    

def test(X, Y):
    y_pred = []
    for i in range(X.shape[0]):
        x = X[i].reshape(len(X[i]), 1)
        if(predict(x) > 0.5):
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred

dataset = pd.read_csv('data.csv')

shuffled_dataset = dataset.sample(frac=1).reset_index(drop=True)

X = shuffled_dataset.iloc[:, 0:-1].values
y = shuffled_dataset.iloc[:, -1].values

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

cv = LeaveOneOut()

y_true, y_pred = list(), list()

for train_ix, test_ix in cv.split(X):
    X_train = X[train_ix, :]
    X_test = X[test_ix, :]
    y_train = y[train_ix]
    y_test = y[test_ix]
    init()
    train(X_train, y_train, epochs)
    yhat = test(X_test, y_test)
    y_true.append(y_test[0])
    y_pred.append(yhat[0])

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(y_true)):
    if(y_true[i] == 1):
        if(y_pred[i] == 1):  
            TP = TP + 1
        else:
            FN = FN + 1

    else:
        if(y_pred[i] == 0):  
            TN = TN + 1
        else:
            FP = FP + 1

accuracy = (TP + TN) / (TP + FP + TN + FN)
precision = (TP) / (TP + FP)
sensitivity = (TP) / (TP + FN)
specificity = (TN) / (FP + TN)
Fmeasure = (2*precision*sensitivity) / (precision + sensitivity)

print([TP, FN], [FP, TN])

print("Accuracy: ", accuracy * 100)
print("Precision: ", precision * 100)
print("Sensitivity: ", sensitivity * 100)
print("Specificity: ", specificity * 100)
print("F-measure: ", Fmeasure)
