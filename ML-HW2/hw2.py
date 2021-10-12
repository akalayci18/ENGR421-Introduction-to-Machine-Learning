import math
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

images_data = pd.read_csv("./hw02_images.csv",header=None)
initialW = pd.read_csv('initial_W.csv',header=None)  
initialW0 = pd.read_csv('initial_w0.csv',header=None) 

one_hot_encoder = OneHotEncoder(categories = 'auto')
labels_encoded = pd.DataFrame(one_hot_encoder.fit_transform((pd.read_csv("./hw02_labels.csv",header=None))).toarray())

trainingset = images_data[0:500]  
training_encoded = labels_encoded[0:500]  

testset = images_data[500:1000]  
test_encoded = labels_encoded[500:1000]   

trainingset = np.array(trainingset)
testset = np.array(testset)

training_encoded = np.array(training_encoded)
test_encoded = np.array(test_encoded)

initialW = np.array(initialW)
initialW0 = np.array(initialW0)


eta = 0.0001
epsilon = 1e-3
max_iteration = 500

def sigmoid(X, w, w0):
    res = 1 / (1 + np.exp(-(np.matmul(X, w) + np.transpose(w0))))
    return res

def gradient_w(X, Y_truth, y_predicted):
    a = (Y_truth - y_predicted)*y_predicted*(1-y_predicted)
    res = -np.matmul(np.transpose(X), np.array(a))
    return res

def gradient_w0(Y_truth, y_predicted):
    res = -np.sum((Y_truth - y_predicted))
    return res




iteration = 1
objective_values = []

while 1:

    y_predicted = sigmoid(trainingset, initialW, initialW0)
    W_old = initialW
    w0_old = initialW0
    initialW = initialW - eta * gradient_w(trainingset, training_encoded, y_predicted)
    initialW0 = initialW0 - eta * gradient_w0(training_encoded, y_predicted)
    objective_values.append(np.sum((training_encoded - y_predicted)**2)*(1/2))
    
    if(np.sqrt(np.sum(initialW0-w0_old)**2 + np.sum(initialW-W_old)**2) < epsilon):
        break
    
    if(iteration >= 500):
        break

    iteration = iteration + 1



plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


print('    y_train \n y_predicted \n', confusion_matrix(pd.DataFrame(y_predicted).values.argmax(axis=1)+1, (pd.read_csv("./hw02_labels.csv",header=None))[0:500]))

print('\n')
new_predicted = sigmoid(testset, initialW, initialW0)

print('    y_train \n y_predicted \n', confusion_matrix(pd.DataFrame(new_predicted).values.argmax(axis=1)+1, (pd.read_csv("./hw02_labels.csv",header=None))[500:1000]))