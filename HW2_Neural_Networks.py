# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 22:05:54 2021

@author: vince
"""


import numpy as np
import pandas as pd
import time
import gc
import random
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import validation_curve
from matplotlib.colors import Normalize
from sklearn.model_selection import StratifiedShuffleSplit
import timeit
import mlrose
import warnings
warnings.filterwarnings("ignore")

class Data():
    
    # points [1]
    def dataAllocation(self,path):
        # Separate out the x_data and y_data and return each
        # args: string path for .csv file
        # return: pandas dataframe, pandas dataframe
        data = pd.read_csv(path)
        xList = [i for i in range(data.shape[1] - 1)]
        x_data = data.iloc[:,xList]
        y_data = data.iloc[:,[-1]]
        # ------------------------------- 
        return x_data,y_data
    
    # points [1]
    def trainSets(self,x_data,y_data):
        # Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
        # Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 614.
        # args: pandas dataframe, pandas dataframe
        # return: pandas dataframe, pandas dataframe, pandas series, pandas series

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=614, shuffle=True)       
        # -------------------------------
        return x_train, x_test, y_train, y_test


#==============================================================================
#Load data
#==============================================================================
datatest = Data()
#path = 'Class_BanknoteAuth.csv'
#path = 'pima-indians-diabetes.csv'
path = 'AFP300_nonAFP300_train_AACandDipeptide_twoSeg.csv'

x_data,y_data = datatest.dataAllocation(path)
print("dataAllocation Function Executed")

x_train, x_test, y_train, y_test = datatest.trainSets(x_data,y_data)
print("trainSets Function Executed")


n = 0
for i in range(y_train.size):
    n = n + y_train.iloc[i,0]
print ('Positive rate for train data is: ',n/y_train.size)

n = 0
for i in range(y_test.size):
    n = n + y_test.iloc[i,0]
print ('Positive rate for test data is: ',n/y_test.size)

#Pre-process the data to standardize it
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#=================================================================================
# Initialize neural network object and fit object
# nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [60], activation = 'sigmoid',
#                                  algorithm = 'random_hill_climb', max_iters = 1000,
#                                  bias = True, is_classifier = True, learning_rate = 0.1,
#                                  early_stopping = True, clip_max = 2000, max_attempts = 300,
#                                  random_state = 3, restarts=10)

# nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [60], activation = 'sigmoid',
#                                  algorithm = 'simulated_annealing', max_iters = 1000,
#                                  bias = True, is_classifier = True, learning_rate = 0.1,
#                                  early_stopping = True, clip_max = 2000, max_attempts = 300,
#                                  random_state = 3, restarts=10)

# nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [60], activation = 'sigmoid',
#                                  algorithm = 'genetic_alg', max_iters = 1000,
#                                  bias = True, is_classifier = True, learning_rate = 0.1,
#                                  early_stopping = True, clip_max = 2000, max_attempts = 300,
#                                  random_state = 3, restarts=10)

# nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
#                                   algorithm = 'random_hill_climb', max_iters = 3000,
#                                   bias = True, is_classifier = True, learning_rate = 10,
#                                   early_stopping = True, clip_max = 1e10, max_attempts = 500,
#                                   random_state = 3, restarts=50)

#schedule = mlrose.ExpDecay(exp_const  = 0.005,init_temp=15)
# schedule = mlrose.ExpDecay(exp_const  = 0.1,init_temp=5)
# nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
#                                   algorithm = 'simulated_annealing', max_iters = 3000,
#                                   bias = True, is_classifier = True, learning_rate = 100,
#                                   early_stopping = True, clip_max = 1e10, max_attempts = 1000,
#                                   random_state = 3, restarts=2, mutation_prob=0.01)

# nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
#                                   algorithm = 'genetic_alg', max_iters = 3000,
#                                   bias = True, is_classifier = True, learning_rate = 10,
#                                   early_stopping = True, clip_max = 2000, max_attempts = 500,
#                                   random_state = 3, restarts=10)


# nn_model1.fit(x_train, y_train)

# # Predict labels for train set and assess accuracy
# y_train_pred = nn_model1.predict(x_train)

# y_train_accuracy = accuracy_score(y_train, y_train_pred)

# print('Training accuracy: ', y_train_accuracy)

# # Predict labels for test set and assess accuracy
# y_test_pred = nn_model1.predict(x_test)

# y_test_accuracy = accuracy_score(y_test, y_test_pred)

# print('Test accuracy: ', y_test_accuracy)

x_train_2, x_validate, y_train_2, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=614, shuffle=True)

#==============================================================================
#Learning rate for randomized hill climbing
y_train_accuracys = []
y_validate_accuracys = []
learning_rates = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

for i in range(len(learning_rates)):
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'random_hill_climb', max_iters = 1000,
                                     bias = True, is_classifier = True, learning_rate = learning_rates[i],
                                     early_stopping = True, clip_max = 1e+10, max_attempts = 300,
                                     random_state = 3, restarts=3)
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
fig = plt.figure()

plt.plot(learning_rates,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(learning_rates,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Learning Rate')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Learning Rate for Randomized Hill Climbing', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_RHC_Accuracy VS Learning Rate for Randomized Hill Climbing.png',dpi=600)

#=========================================================================================================
y_train_accuracys = []
y_validate_accuracys = []
restarts = [1, 5, 10, 15, 25, 40, 50]

for i in range(len(restarts)):
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'random_hill_climb', max_iters = 3000,
                                     bias = True, is_classifier = True, learning_rate = 10,
                                     early_stopping = True, clip_max = 1e+10, max_attempts = 500,
                                     random_state = 3, restarts=restarts[i])
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
fig = plt.figure()

plt.plot(restarts,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(restarts,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Number of Restarts')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Number of Restarts for Randomized Hill Climbing', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
#plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_RHC_Accuracy VS Number of Restarts for Randomized Hill Climbing.png',dpi=600)

#=========================================================================================================
y_train_accuracys = []
y_validate_accuracys = []
max_iters = [250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000,10000,15000,20000,30000]

for i in range(len(max_iters)):
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'random_hill_climb', max_iters = max_iters[i],
                                     bias = True, is_classifier = True, learning_rate = 100,
                                     early_stopping = True, clip_max = 1e+10, max_attempts = 500,
                                     random_state = 3, restarts=25)
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
fig = plt.figure()

plt.plot(max_iters,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(max_iters,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Max Iterations')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Max Iterations for Random Hill Climbing', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
#plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_RHC_Accuracy VS Max Iterations for Random Hill Climbing.png',dpi=600)

#=======================================================================================================
y_train_accuracys = []
y_validate_accuracys = []
clip_maxs  = [100, 1000, 10000, 1e5, 1e6,1e7]
losses = []
fitness_curves = []
iterations = []

for i in range(len(clip_maxs)):
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'random_hill_climb', max_iters = 15000,
                                     bias = True, is_classifier = True, learning_rate = 100,
                                     early_stopping = True, clip_max = clip_maxs[i], max_attempts = 500,
                                     random_state = 3, restarts=25, curve =True)
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
    losses.append(nn_model1.loss)
    fitness_curves.append(nn_model1.fitness_curve)
    iterations.append(len(nn_model1.fitness_curve))
    
fig = plt.figure()

plt.plot(clip_maxs,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(clip_maxs,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Maximum Weight')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Maximum Weight for Random Hill Climbing', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_RHC_Accuracy VS Maximum Weight for Random Hill Climbing.png',dpi=600)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(clip_maxs,losses, label = 'Loss Value', color="C0", lw=2)
ax1.set_ylabel('Loss Value')
ax1.set_title("Effect of Maximum Weight for Random Hill Climbing")
ax1.set_xlabel('Maximum Weight')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(clip_maxs, iterations, label = 'Number of Iterations', color="C1", lw=2)
#ax2.set_xlim([0, np.e])
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(0.5,1), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig('NN_RHC_Effect of Maximum Weight for Random Hill Climbing.png',dpi=600)

#========================================================================================================
#Learning rate for simulated annealing
y_train_accuracys = []
y_validate_accuracys = []
learning_rates = [0.001, 0.01, 0.1, 1, 10, 100, 1000,10000]
losses = []
fitness_curves = []
iterations = []
fitted_weights = []

schedule = mlrose.GeomDecay()
for i in range(len(learning_rates)):
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'simulated_annealing', max_iters = 1000,
                                     bias = True, is_classifier = True, learning_rate = learning_rates[i],
                                     early_stopping = True, clip_max = 1e+10, max_attempts = 300,
                                     random_state = 3, schedule=schedule, curve =True)
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
    losses.append(nn_model1.loss)
    fitness_curves.append(nn_model1.fitness_curve)
    iterations.append(len(nn_model1.fitness_curve))
    fitted_weights.append(nn_model1.fitted_weights)
    
fig = plt.figure()

plt.plot(learning_rates,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(learning_rates,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Learning Rate')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Learning Rate for Simulated Annealing', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_SA_Accuracy VS Learning Rate for Simulated Annealing.png',dpi=600)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(learning_rates,losses, label = 'Loss Value', color="C0", lw=2)
ax1.set_ylabel('Loss Value')
ax1.set_title("Effect of Learning Rate for Simulated Annealing")
ax1.set_xlabel('Learning Rate')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(learning_rates, iterations, label = 'Number of Iterations', color="C1", lw=2)
#ax2.set_xlim([0, np.e])
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.xscale('log')
plt.savefig('NN_SA_Effect of Learning Rate for Simulated Annealing.png',dpi=600)

#============================================================================================
#initial temperature
y_train_accuracys = []
y_validate_accuracys = []
init_temps = [0.01, 0.1, 0.5, 1, 5, 10]


for i in range(len(init_temps)):
    schedule = mlrose.GeomDecay(init_temp=init_temps[i])
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'simulated_annealing', max_iters = 1000,
                                     bias = True, is_classifier = True, learning_rate = 100,
                                     early_stopping = True, clip_max = 1e+10, max_attempts = 300,
                                     random_state = 3, schedule=schedule)
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
fig = plt.figure()

plt.plot(init_temps,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(init_temps,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Initial Temperature ')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Initial Temperature for Simulated Annealing', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
#plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_SA_Accuracy VS Initial Temperature for Simulated Annealing.png',dpi=600)

#=======================================================================================================
#decay constant
y_train_accuracys = []
y_validate_accuracys = []
decays = [0.8, 0.85, 0.9, 0.95, 0.99, 0.999]


for i in range(len(decays)):
    schedule = mlrose.GeomDecay(init_temp=0.01,decay=decays[i])
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'simulated_annealing', max_iters = 1000,
                                     bias = True, is_classifier = True, learning_rate = 100,
                                     early_stopping = True, clip_max = 1e+10, max_attempts = 300,
                                     random_state = 3, schedule=schedule)
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
fig = plt.figure()

plt.plot(decays,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(decays,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Temperature Decay Parameter')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Temperature Decay Parameter for Simulated Annealing', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
#plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_SA_Accuracy VS Temperature Decay Parameter for Simulated Annealing.png',dpi=600)

#====================================================================================================
#max iteration
y_train_accuracys = []
y_validate_accuracys = []
max_iters = [250, 500, 1000, 2000, 3000, 4000, 5000, 6000,8000,10000,15000,20000,30000,40000]
losses = []
fitness_curves = []
iterations = []
fitted_weights = []


for i in range(len(max_iters)):
    schedule = mlrose.GeomDecay(init_temp=0.01,decay=0.9, min_temp=0.00001)
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'simulated_annealing', max_iters = max_iters[i],
                                     bias = True, is_classifier = True, learning_rate = 100,
                                     early_stopping = True, clip_max = 1e+10, max_attempts = 500,
                                     random_state = 3, schedule=schedule,curve=True)
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
    losses.append(nn_model1.loss)
    fitness_curves.append(nn_model1.fitness_curve)
    iterations.append(len(nn_model1.fitness_curve))
    fitted_weights.append(nn_model1.fitted_weights)
    
fig = plt.figure()

plt.plot(max_iters,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(max_iters,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Max Iteration')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Max Iteration for Simulated Annealing', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
#plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_SA_Accuracy VS Max Iteration for Simulated Annealing.png',dpi=600)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(max_iters,losses, label = 'Loss Value', color="C0", lw=2)
ax1.set_ylabel('Loss Value')
ax1.set_title("Effect of Max Iteration for Simulated Annealing")
ax1.set_xlabel('Max Iteration')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(max_iters, iterations, label = 'Number of Iterations', color="C1", lw=2)
#ax2.set_xlim([0, np.e])
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.tight_layout()
#plt.xscale('log')
plt.savefig('NN_SA_Effect of Max Iteration for Simulated Annealing.png',dpi=600)
#======================================================================================================
#Learning rate for Gradient Descent
y_train_accuracys = []
y_validate_accuracys = []
learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

for i in range(len(learning_rates)):
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'gradient_descent', max_iters = 2000,
                                     bias = True, is_classifier = True, learning_rate = learning_rates[i],
                                     early_stopping = True, clip_max = 1e+10, max_attempts = 300,
                                     random_state = 3, restarts=3)
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
fig = plt.figure()

plt.plot(learning_rates,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(learning_rates,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Learning Rate')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Learning Rate for Gradient Descent', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_GD_Accuracy VS Learning Rate for Gradient Descent.png',dpi=600)

#======================================================================================================
#Max iteration for Gradient Descent
y_train_accuracys = []
y_validate_accuracys = []
max_iters = [50, 100, 200, 300, 400, 500, 1000, 1200, 1300, 1400, 1500,1600,1700,1800, 2000, 2500, 3000]

for i in range(len(max_iters)):
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'gradient_descent', max_iters = max_iters[i],
                                     bias = True, is_classifier = True, learning_rate = 3e-4,
                                     early_stopping = True, clip_max = 1e+10, max_attempts = 300,
                                     random_state = 3, restarts=3)
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
fig = plt.figure()

plt.plot(max_iters,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(max_iters,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Max Iteration')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Max Iteration for Gradient Descent', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
#plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_GD_Accuracy VS Max Iteration for Gradient Descent.png',dpi=600)

#==================================================================================
#clip_maxs for Gradient Descent
y_train_accuracys = []
y_validate_accuracys = []
clip_maxs  = [10, 100, 1000, 10000, 1e5, 1e6,1e7]

for i in range(len(clip_maxs)):
    nn_model1 = nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'gradient_descent', max_iters = 1000,
                                     bias = True, is_classifier = True, learning_rate = 3e-4,
                                     early_stopping = True, clip_max = clip_maxs[i], max_attempts = 300,
                                     random_state = 3, restarts=3)
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
fig = plt.figure()

plt.plot(clip_maxs,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(clip_maxs,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Maximum Weight')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Maximum Weight for Gradient Descent', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_GD_Accuracy VS Maximum Weight for Gradient Descentm.png',dpi=600)


#==============================================================================
#Learning rate for Generic Algorithm
y_train_accuracys = []
y_validate_accuracys = []
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

for i in range(len(learning_rates)):
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'genetic_alg', max_iters = 2000,
                                     bias = True, is_classifier = True, learning_rate = learning_rates[i],
                                     early_stopping = True, clip_max = 1e+10, max_attempts = 100,
                                     random_state = 3, restarts=3)
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
fig = plt.figure()

plt.plot(learning_rates,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(learning_rates,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Learning Rate')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Learning Rate for Genetic Algorithm', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_GA_Accuracy VS Learning Rate for Genetic Algorithm.png',dpi=600)
#================================================================================================
y_train_accuracys = []
y_validate_accuracys = []
pop_sizes  = [50, 100, 200, 500, 1000, 1500]

for i in range(len(pop_sizes)):
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'genetic_alg', max_iters = 2000,
                                     bias = True, is_classifier = True, learning_rate = 10,
                                     early_stopping = True, clip_max = 1e+10, max_attempts = 100,
                                     random_state = 3,pop_size=pop_sizes[i])
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
fig = plt.figure()

plt.plot(pop_sizes,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(pop_sizes,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Size of Population')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Size of Population for Genetic Algorithm', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
#plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_GA_Accuracy VS Size of Population for Genetic Algorithm.png',dpi=600)
#================================================================================================
y_train_accuracys = []
y_validate_accuracys = []
mutation_probs  = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9,0.95]

for i in range(len(mutation_probs)):
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'genetic_alg', max_iters = 2000,
                                     bias = True, is_classifier = True, learning_rate = 10,
                                     early_stopping = True, clip_max = 1e+10, max_attempts = 100,
                                     random_state = 3,pop_size=200,mutation_prob=mutation_probs[i])
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
fig = plt.figure()

plt.plot(mutation_probs,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(mutation_probs,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Probability of Mutation')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Probability of Mutation for Genetic Algorithm', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
#plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_GA_Accuracy VS Probability of Mutation for Genetic Algorithm.png',dpi=600)

#================================================================================================
y_train_accuracys = []
y_validate_accuracys = []
clip_maxs  = [10, 100, 1000, 10000, 1e5, 1e6,1e7]

for i in range(len(clip_maxs)):
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'genetic_alg', max_iters = 2000,
                                     bias = True, is_classifier = True, learning_rate = 10,
                                     early_stopping = True, clip_max = clip_maxs[i], max_attempts = 100,
                                     random_state = 3,pop_size=200,mutation_prob=0.1)
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
fig = plt.figure()

plt.plot(clip_maxs,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(clip_maxs,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Maximum Weight')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Maximum Weight for Genetic Algorithm', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_GA_Accuracy VS Maximum Weight for Genetic Algorithm.png',dpi=600)

#================================================================================================
y_train_accuracys = []
y_validate_accuracys = []
max_iters = [5, 10, 15, 20, 25, 50, 75,100,150,200]
losses = []
fitness_curves = []
iterations = []
fitted_weights = []

for i in range(len(max_iters)):
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'genetic_alg', max_iters = max_iters[i],
                                     bias = True, is_classifier = True, learning_rate = 10,
                                     early_stopping = True, clip_max = 1e10, max_attempts = 100,
                                     random_state = 3,pop_size=200,mutation_prob=0.1,curve=True)
    
    nn_model1.fit(x_train_2, y_train_2)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(x_train_2)    
    y_train_accuracy = accuracy_score(y_train_2, y_train_pred)
    y_train_accuracys.append(y_train_accuracy) 
    # Predict labels for test set and assess accuracy
    y_validate_pred = nn_model1.predict(x_validate)    
    y_validate_accuracy = accuracy_score(y_validate, y_validate_pred)
    y_validate_accuracys.append(y_validate_accuracy)
    
    losses.append(nn_model1.loss)
    fitness_curves.append(nn_model1.fitness_curve)
    iterations.append(len(nn_model1.fitness_curve))
    fitted_weights.append(nn_model1.fitted_weights)
    
fig = plt.figure()

plt.plot(max_iters,y_train_accuracys, label = 'Accuracy for training data', color="C0", lw=2)
plt.plot(max_iters,y_validate_accuracys, label = 'Accuracy for validation data', color="C1", lw=2)


plt.ylabel('Accuracy')
plt.xlabel('Max Iteration')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Accuracy VS Max Iteration for Genetic Algorithm', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
#plt.xscale('log')
plt.tight_layout()
plt.savefig('NN_GA_Accuracy VS Max Iteration for Genetic Algorithm.png',dpi=600)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(max_iters,losses, label = 'Loss Value', color="C0", lw=2)
ax1.set_ylabel('Loss Value')
ax1.set_title("Effect of Max Iteration for Genetic Algorithm")
ax1.set_xlabel('Max Iteration')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(max_iters, iterations, label = 'Number of Iterations', color="C1", lw=2)
#ax2.set_xlim([0, np.e])
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.tight_layout()
#plt.xscale('log')
plt.savefig('NN_GA_Effect of Max Iteration for Genetic Algorithm.png',dpi=600)


#=========================================================================================
start_rhc = time.time()
nn_model_rhc = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                 algorithm = 'random_hill_climb', max_iters = 30000,
                                 bias = True, is_classifier = True, learning_rate = 100,
                                 early_stopping = True, clip_max = 1e+10, max_attempts = 500,
                                 random_state = 3, restarts=25, curve=True)
    
nn_model_rhc.fit(x_train, y_train)
end_rhc = time.time()
    
# Predict labels for train set and assess accuracy
y_train_pred = nn_model_rhc.predict(x_train)    
y_train_accuracy = accuracy_score(y_train, y_train_pred) 
# Predict labels for test set and assess accuracy
y_test_pred = nn_model_rhc.predict(x_test)    
y_test_accuracy = accuracy_score(y_test, y_test_pred)

print('#'*50)
print('random hill climb')
print('The fitness at the best state is: ', nn_model_rhc.loss)
print('Optimization time is: ',end_rhc - start_rhc)
print('Training accuracy is : ',y_train_accuracy)
print('Test accuracy is : ',y_test_accuracy)

#=========================================================================================
start_sa = time.time()
schedule = mlrose.GeomDecay(init_temp=0.01,decay=0.9, min_temp=0.00001)
nn_model_sa = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                 algorithm = 'simulated_annealing', max_iters = 40000,
                                 bias = True, is_classifier = True, learning_rate = 100,
                                 early_stopping = True, clip_max = 1e+10, max_attempts = 500,
                                 random_state = 3, schedule=schedule,curve=True)
    
nn_model_sa.fit(x_train, y_train)
end_sa = time.time()
    
# Predict labels for train set and assess accuracy
y_train_pred = nn_model_sa.predict(x_train)    
y_train_accuracy = accuracy_score(y_train, y_train_pred) 
# Predict labels for test set and assess accuracy
y_test_pred = nn_model_sa.predict(x_test)    
y_test_accuracy = accuracy_score(y_test, y_test_pred)

print('#'*50)
print('simulated annealing')
print('The fitness at the best state is: ', nn_model_sa.loss)
print('Optimization time is: ',end_sa - start_sa)
print('Training accuracy is : ',y_train_accuracy)
print('Test accuracy is : ',y_test_accuracy)

#=========================================================================================
start_ga = time.time()
nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                   algorithm = 'genetic_alg', max_iters = 100,
                                   bias = True, is_classifier = True, learning_rate = 10,
                                   early_stopping = True, clip_max = 1e10, max_attempts = 100,
                                   random_state = 3,pop_size=200,mutation_prob=0.1,curve=True)
    
nn_model_ga.fit(x_train, y_train)
end_ga = time.time()
    
# Predict labels for train set and assess accuracy
y_train_pred = nn_model_ga.predict(x_train)    
y_train_accuracy = accuracy_score(y_train, y_train_pred) 
# Predict labels for test set and assess accuracy
y_test_pred = nn_model_ga.predict(x_test)    
y_test_accuracy = accuracy_score(y_test, y_test_pred)

print('#'*50)
print('genetic annealing')
print('The fitness at the best state is: ', nn_model_ga.loss)
print('Optimization time is: ',end_ga - start_ga)
print('Training accuracy is : ',y_train_accuracy)
print('Test accuracy is : ',y_test_accuracy)

#=========================================================================================
start_gd = time.time()
nn_model_gd = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'sigmoid',
                                     algorithm = 'gradient_descent', max_iters = 1000,
                                     bias = True, is_classifier = True, learning_rate = 3e-4,
                                     early_stopping = True, clip_max = 1e+10, max_attempts = 300,
                                     random_state = 3, restarts=3,curve=True)
    
nn_model_gd.fit(x_train, y_train)
end_gd = time.time()
    
# Predict labels for train set and assess accuracy
y_train_pred = nn_model_gd.predict(x_train)    
y_train_accuracy = accuracy_score(y_train, y_train_pred) 
# Predict labels for test set and assess accuracy
y_test_pred = nn_model_gd.predict(x_test)    
y_test_accuracy = accuracy_score(y_test, y_test_pred)

print('#'*50)
print('gradient decent')
print('The fitness at the best state is: ', nn_model_gd.loss)
print('Optimization time is: ',end_gd - start_gd)
print('Training accuracy is : ',y_train_accuracy)
print('Test accuracy is : ',y_test_accuracy)

#==========================================================================================
fig = plt.figure()

plt.plot(-nn_model_rhc.fitness_curve, label = 'Randomized hill climbing', color="C0", lw=2)
plt.plot(-nn_model_sa.fitness_curve, label = 'Simulated annealing', color="C1", lw=2)
plt.plot(-nn_model_ga.fitness_curve, label = 'Genetic algorithm', color="C2", lw=2)
plt.plot(-nn_model_gd.fitness_curve, label = 'Gradient_descent', color="C3", lw=2)


plt.ylabel('Fitness Value (Loss)')
plt.xlabel('Iteration')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Fitness Value for Weights of Neural Netword VS Iteration', y = 1.03)
plt.legend(loc=0)
#plt.ylim(0,40)
plt.grid(True)
plt.xscale('log')
#plt.yscale('log')
plt.tight_layout()
plt.savefig('NN_Fitness Value VS Iteration 3.png',dpi=600)
