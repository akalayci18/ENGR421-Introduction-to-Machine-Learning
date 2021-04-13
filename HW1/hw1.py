import math
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


images_data = pd.read_csv("./hw01_images.csv",header=None)
labels_data = pd.read_csv("./hw01_labels.csv",header=None)

trainingset_x = images_data.iloc[0:200]
trainingset_y = labels_data.iloc[0:200]

testset_x = images_data.iloc[200:400]
testset_y = labels_data.iloc[200:400]

py1= trainingset_x[trainingset_y[0]==1]
py2= trainingset_x[trainingset_y[0]==2]

means =  np.array([py1.mean(), py2.mean()])

print("print(means[:,0]) \n", means[0])
print("print(means[:,1]) \n", means[1])

deviations = np.array([np.sqrt(np.mean(((py1-means[0])**2))), np.sqrt(np.mean(((py2-means[1]))**2))])

print("print(deviations[:,0]) \n", deviations[0])
print("print(deviations[:,1]) \n", deviations[1])


length_py1 = len(py1)
length_py2 = len(py2)

priors = length_py1 / 200, length_py2 / 200
print("print(priors) \n", priors)


def res(t_set):
    
    score_func1 = sum((-0.5*np.log(2*math.pi)) - (np.log(deviations[0])) - ((t_set-means[0])*(t_set-means[0])/(2*deviations[0]*deviations[0]))) + np.log(priors[0]) 
    score_func2 = sum((-0.5*np.log(2*math.pi)) - (np.log(deviations[1])) - ((t_set-means[1])*(t_set-means[1])/(2*deviations[1]*deviations[1]))) + np.log(priors[1])
 

    if score_func2  < score_func1 :
        return 1
    else:
        return 2
    

y_train = trainingset_x.apply(lambda t_set: res(t_set), axis=1)
y_test = testset_x.apply(lambda t_set: res(t_set), axis=1)

print('    y_hat \n y_train \n', confusion_matrix(trainingset_y, y_train))
print('\n')
print('    y_hat \n y_test \n', confusion_matrix(trainingset_y, y_test))

