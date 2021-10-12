import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


data_set = pd.read_csv("hw03_data_set.csv")

y = data_set.iloc[:,1]
X = data_set.iloc[:,0]

trainingset_x = data_set.iloc[:150,0]
testset_x = data_set.iloc[150:,0]
trainingset_y = data_set.iloc[:150,1]
testset_y = data_set.iloc[150:,1]


trainingset_x = trainingset_x.values
trainingset_y = trainingset_y.values

testset_y = testset_y.values
testset_x = testset_x.values


bin_width = 0.37
origin = 1.5

left_borders = np.arange(origin, max(trainingset_x),bin_width)
data_interval = np.arange(origin,max(trainingset_x),0.001)
right_borders = np.arange(origin + bin_width,max(trainingset_x)+bin_width,bin_width)



def regressogram(val):
    total = 0
    count = 0
    length_x = len(trainingset_x)
    
    for i in range(0,length_x):
        smaller = trainingset_x[i] <= right_borders.T[val]
        bigger = left_borders.T[val] < trainingset_x[i]
        if(smaller):
            if(bigger):
                total = total +  trainingset_y[i]
                count = count + 1
    return total/count


def running_mean_smoother(val):
    total = 0
    count = 0
    bin_width = 0.37
    length_x = len(trainingset_x)
    for i in range(0,length_x):
        sub = data_interval[val]-trainingset_x[i]
        if(np.abs(sub/bin_width) < 0.5):
            total = total +  trainingset_y[i]
            count = count + 1
    return total/count


def kernel_smoother(val):
    total = 0
    count = 0
    bin_width = 0.37
    length_x = len(trainingset_x)
    for i in range(0,length_x):
        eq = 1/np.sqrt(2* math.pi) * np.exp(-1* ((data_interval[val] - trainingset_x[i]) / bin_width) ** 2 / 2)
        count = count + eq
        total = total + (eq * trainingset_y[i])
        
    return total/count


def error(testset_y,testset_x,p_h_type,type):
    calc_rmse = 0
    lengthset_x = len(testset_x)
    for i in range(0,lengthset_x):
            if type == "Regressogram":
                for j in range(0,len(left_borders)):
                    smaller=left_borders[j] < testset_x[i] and testset_x[i] <= right_borders[j]
                    if(smaller):
                        calc_rmse = calc_rmse + (testset_y[i] - p_h_type[int((testset_x[i]-origin)/bin_width)])**2
                res = np.sqrt(calc_rmse /lengthset_x)
            elif type == "Kernel Smoother":
                calc_rmse = calc_rmse + (testset_y[i] - p_h_type[int((testset_x[i]-origin)/0.001)])**2
                div = np.sqrt(calc_rmse /lengthset_x)
                res = div
            elif type == "Running Mean Smoother":
                calc_rmse = calc_rmse + (testset_y[i] - p_h_type[int((testset_x[i]-origin)/0.001)])**2
                div = np.sqrt(calc_rmse / lengthset_x)
                res = np.sqrt(calc_rmse / lengthset_x)
    print("{type} => RMSE is {calc_rmse} when h is {h}".format(type = type,calc_rmse = np.round(res,4), h = bin_width))


p_head = []
length_lb=len(left_borders)

for i in range(0,length_lb):
    p_head.append(regressogram(i)) 

p_head = np.array(p_head)


plt.scatter(trainingset_x,trainingset_y, color='blue', s=20, alpha=0.8)
plt.scatter(testset_x,testset_y, color='red', s=20, alpha=0.8)
plt.legend(['training','test'])

for i in range(1,len(left_borders)+1):
    plt.plot((left_borders[i-1],right_borders[i-1]),(p_head[i-1],p_head[i-1]),color = 'black')
    if i < len(left_borders):
        plt.plot((right_borders[i-1],right_borders[i-1]),(p_head[i-1],p_head[i]),color = 'black')


plt.ylim = ((min(y), max(y)))
plt.xlim = ((min(trainingset_x)  + bin_width, max(trainingset_x) + bin_width ))
plt.show()

error(testset_y,testset_x,p_h_type = p_head,type = "Regressogram")

p_head_runmean = []
for i in range(0,len(data_interval)):
    p_head_runmean.append(running_mean_smoother(i))  
p_head_runmean = np.array(p_head_runmean)

plt.scatter(trainingset_x,trainingset_y, color='blue', s=20, alpha=0.8)
plt.scatter(testset_x,testset_y, color='red', s=20, alpha=0.8)
plt.legend(['training','test'])

plt.ylim = ((min(y), max(y)))
plt.xlim = ((min(trainingset_x), max(trainingset_x)))

plt.plot(data_interval,p_head_runmean, color = 'black')
plt.show()

error(testset_y,testset_x,p_h_type = p_head_runmean,type = "Running Mean Smoother")

p_head_kernel = []
for i in range(0,len(data_interval)):
    p_head_kernel.append(kernel_smoother(i))  
p_head_kernel = np.array(p_head_kernel)


plt.scatter(trainingset_x,trainingset_y, color='blue', s=20, alpha=0.8)
plt.scatter(testset_x,testset_y, color='red', s=20, alpha=0.8)
plt.legend(['training','test'])


plt.ylim = ((min(y), max(y)))
plt.xlim = ((min(trainingset_x)-bin_width, max(trainingset_x)+bin_width ))
plt.plot(data_interval,p_head_kernel,color = 'black')
plt.show()

error(testset_y,testset_x,p_h_type = p_head_kernel,type = "Kernel Smoother")