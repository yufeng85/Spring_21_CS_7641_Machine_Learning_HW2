# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:49:01 2021

@author: vince
"""

import mlrose
import numpy as np
import matplotlib.pyplot as plt
import timeit
import time

# Define fitness function
size = 100
fitness = mlrose.OneMax()

# Define problem object
problem = mlrose.DiscreteOpt(length = size, fitness_fn = fitness,
                             maximize = True, max_val = 2)
# Define initial state
#init_state = np.zeros((size,), dtype=int)
np.random.seed(seed=1)
init_state = np.random.randint(0, 2, size)

#================================================================================================
# Solve problem using random hill climb
best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem,
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state, restarts=4, curve=True, random_state = 1)
print('#'*50)
print('random hill climb')
print('The best state found is: ', best_state_rhc)
print('The fitness at the best state is: ', best_fitness_rhc)

fig = plt.figure()
plt.plot(fitness_curve_rhc, label = 'Fitness Value', color="navy", lw=2)
plt.ylabel('Fitness Value')
plt.xlabel('Iterations')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Fitness Value VS Iterations (Randomized Hill Climbing)', y = 1.03)
#plt.legend()
plt.grid(True)
plt.savefig('OneMax_RHC_Fitness Value VS Iterations.png',dpi=600)

# Solve problem using simulated annealing
# Define decay schedule
schedule = mlrose.ExpDecay()

best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedule,
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state, curve=True, random_state = 1)

print('#'*50)
print('simulated annealing')
print('The best state found is: ', best_state_sa)
print('The fitness at the best state is: ', best_fitness_sa)

fig = plt.figure()
plt.plot(fitness_curve_sa, label = 'Fitness Value', color="navy", lw=2)
plt.ylabel('Fitness Value')
plt.xlabel('Iterations')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Fitness Value VS Iterations (Simulated Annealing)', y = 1.03)
#plt.legend()
plt.grid(True)
plt.savefig('OneMax_SA_Fitness Value VS Iterations.png',dpi=600)

# Solve problem using genetic algorithm
best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem,
                                                      pop_size=500, mutation_prob=0.1,                                        
                                                      max_attempts = 100, max_iters = 1000,
                                                      curve=True, random_state = 1)

print('#'*50)
print('genetic algorithm')
print('The best state found is: ', best_state_ga)
print('The fitness at the best state is: ', best_fitness_ga)

fig = plt.figure()
plt.plot(fitness_curve_ga, label = 'Fitness Value', color="navy", lw=2)
plt.ylabel('Fitness Value')
plt.xlabel('Iterations')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Fitness Value VS Iterations (Genetic Algorithm)', y = 1.03)
#plt.legend()
plt.grid(True)
plt.savefig('OneMax_GA_Fitness Value VS Iterations.png',dpi=600)

# Solve problem using MIMIC
best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem,
                                                      pop_size=200, keep_pct=0.1,                                        
                                                      max_attempts = 10, max_iters = 1000,
                                                      curve=True, random_state = 1)

print('#'*50)
print('MIMIC')
print('The best state found is: ', best_state_mimic)
print('The fitness at the best state is: ', best_fitness_mimic)

fig = plt.figure()
plt.plot(fitness_curve_mimic, label = 'Fitness Value', color="navy", lw=2)
plt.ylabel('Fitness Value')
plt.xlabel('Iterations')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Fitness Value VS Iterations (MIMIC)', y = 1.03)
#plt.legend()
plt.grid(True)
plt.savefig('OneMax_MIMIC_Fitness Value VS Iterations.png',dpi=600)

#================================================================================================
# Solve problem using random hill climb
restarts = [0,1,2,3,4,5,6,7,8,9,10,15,20,25,30]
fitness_curve_rhcs = []
best_state_rhcs = []
best_fitness_rhcs = []
iteration_rhc = []
for i in range(len(restarts)):
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem,
                                                      max_attempts = 25, max_iters = 1000,
                                                      init_state = init_state, restarts=restarts[i], curve=True, random_state = 1)
    best_state_rhcs.append(best_state_rhc)
    best_fitness_rhcs.append(best_fitness_rhc)
    fitness_curve_rhcs.append(fitness_curve_rhc)
    iteration_rhc.append(len(fitness_curve_rhc))

fig = plt.figure()
plt.plot(restarts, best_fitness_rhcs, label = 'Fitness Value', color="navy", lw=2)
plt.ylabel('Fitness Value')
plt.xlabel('Number of random restarts')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Fitness Value VS Number of Random Restarts (Randomized Hill Climbing)', y = 1.03)
#plt.legend()
#plt.ylim(0,40)
plt.grid(True)
plt.savefig('OneMax_RHC_Fitness Value VS Number of Random Restarts.png',dpi=600)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(restarts,best_fitness_rhcs, label = 'Fitness Value', color="C0", lw=2)
ax1.set_ylabel('Fitness Value')
ax1.set_title("Effect of Number of Random Restarts for Randomized Hill Climbing")
ax1.set_xlabel('Number of random restarts')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(restarts, iteration_rhc, label = 'Number of Iterations', color="C1", lw=2)
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig('OneMax_RHC_Effect of Number of Random Restarts for Randomized Hill Climbing.png',dpi=600)

#================================================================================================
# Solve problem using simulated annealing
#=============================================================================================
init_temps = [1,5,10,15,20,25,30,35,40,45,50,55,60]
fitness_curve_sas = []
best_state_sas = []
best_fitness_sas = []
iteration_sa = []
for i in range(len(init_temps)):
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = mlrose.GeomDecay(init_temps[i]),
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state, curve=True, random_state = 1)
    best_state_sas.append(best_state_sa)
    best_fitness_sas.append(best_fitness_sa)
    fitness_curve_sas.append(fitness_curve_sa)
    iteration_sa.append(len(fitness_curve_sa))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(init_temps,best_fitness_sas, label = 'Fitness Value', color="C0", lw=2)
ax1.set_ylabel('Fitness Value')
ax1.set_title("Effect of Initial Temperature for GeomDecay")
ax1.set_xlabel('Initial Temperature')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(init_temps, iteration_sa, label = 'Number of Iterations', color="C1", lw=2)
#ax2.set_xlim([0, np.e])
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(0.5,0.2), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig('OneMax_SA_Effect of Initial Temperature for GeomDecay.png',dpi=600)

#=============================================================================================
# decays = [0.94,0.96,0.98,0.99,0.995,0.999]
decays = [0.8,0.84,0.88,0.92,0.96,0.99]
fitness_curve_sas = []
best_state_sas = []
best_fitness_sas = []
iteration_sa = []
for i in range(len(decays)):
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = mlrose.GeomDecay(decay=decays[i],init_temp =2),
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state, curve=True, random_state = 1)
    best_state_sas.append(best_state_sa)
    best_fitness_sas.append(best_fitness_sa)
    fitness_curve_sas.append(fitness_curve_sa)
    iteration_sa.append(len(fitness_curve_sa))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(decays,best_fitness_sas, label = 'Fitness Value', color="C0", lw=2)
ax1.set_ylabel('Fitness Value')
ax1.set_title("Effect of Temperature decay parameter for GeomDecay")
ax1.set_xlabel('Temperature Decay Parameter ')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(decays, iteration_sa, label = 'Number of Iterations', color="C1", lw=2)
#ax2.set_xlim([0, np.e])
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(0.5,0.2), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig('OneMax_SA_Effect of Temperature decay parameter for GeomDecay.png',dpi=600)

#=============================================================================================
init_temps = [1,5,10,15,20,25,30,35,40,45,50,55,60]
fitness_curve_sas = []
best_state_sas = []
best_fitness_sas = []
iteration_sa = []
for i in range(len(init_temps)):
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = mlrose.ExpDecay(init_temps[i]),
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state, curve=True, random_state = 1)
    best_state_sas.append(best_state_sa)
    best_fitness_sas.append(best_fitness_sa)
    fitness_curve_sas.append(fitness_curve_sa)
    iteration_sa.append(len(fitness_curve_sa))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(init_temps,best_fitness_sas, label = 'Fitness Value', color="C0", lw=2)
ax1.set_ylabel('Fitness Value')
ax1.set_title("Effect of Initial Temperature for ExpDecay")
ax1.set_xlabel('Initial Temperature')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(init_temps, iteration_sa, label = 'Number of Iterations', color="C1", lw=2)
#ax2.set_xlim([0, np.e])
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(0.5,0.2), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig('OneMax_SA_Effect of Initial Temperature for ExpDecay.png',dpi=600)

#=============================================================================================
exp_consts  = [0.001,0.002,0.003,0.005,0.015,0.02,0.025,0.03]
fitness_curve_sas = []
best_state_sas = []
best_fitness_sas = []
iteration_sa = []
for i in range(len(exp_consts)):
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = mlrose.ExpDecay(exp_const=exp_consts[i],init_temp =10),
                                                      max_attempts = 58, max_iters = 1000,
                                                      init_state = init_state, curve=True, random_state = 1)
    best_state_sas.append(best_state_sa)
    best_fitness_sas.append(best_fitness_sa)
    fitness_curve_sas.append(fitness_curve_sa)
    iteration_sa.append(len(fitness_curve_sa))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(exp_consts,best_fitness_sas, label = 'Fitness Value', color="C0", lw=2)
ax1.set_ylabel('Fitness Value')
ax1.set_title("Effect of Exponential Constant Parameter for ExpDecay")
ax1.set_xlabel('Exponential Constant Parameter')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(exp_consts, iteration_sa, label = 'Number of Iterations', color="C1", lw=2)
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(0.5,0.2), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig('OneMax_SA_Effect of Exponential Constant Parameter for ExpDecay.png',dpi=600)

#==============================================================================================
init_temps = [1,5,10,15,20,25,30,35,40,45,50,55,60]
fitness_curve_sas = []
best_state_sas = []
best_fitness_sas = []
iteration_sa = []
for i in range(len(init_temps)):
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = mlrose.ArithDecay(init_temps[i]),
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state, curve=True, random_state = 1)
    best_state_sas.append(best_state_sa)
    best_fitness_sas.append(best_fitness_sa)
    fitness_curve_sas.append(fitness_curve_sa)
    iteration_sa.append(len(fitness_curve_sa))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(init_temps,best_fitness_sas, label = 'Fitness Value', color="C0", lw=2)
ax1.set_ylabel('Fitness Value')
ax1.set_title("Effect of Initial Temperature for ArithDecay")
ax1.set_xlabel('Initial Temperature')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(init_temps, iteration_sa, label = 'Number of Iterations', color="C1", lw=2)
#ax2.set_xlim([0, np.e])
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig('OneMax_SA_Effect of Initial Temperature for ArithDecay.png',dpi=600)

#=============================================================================================
decays   = [0.001,0.002,0.003,0.005,0.015,0.02,0.025,0.03]
fitness_curve_sas = []
best_state_sas = []
best_fitness_sas = []
iteration_sa = []
for i in range(len(decays)):
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = mlrose.ArithDecay(decay=decays[i],init_temp =1),
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state, curve=True, random_state = 1)
    best_state_sas.append(best_state_sa)
    best_fitness_sas.append(best_fitness_sa)
    fitness_curve_sas.append(fitness_curve_sa)
    iteration_sa.append(len(fitness_curve_sa))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(exp_consts,best_fitness_sas, label = 'Fitness Value', color="C0", lw=2)
ax1.set_ylabel('Fitness Value')
ax1.set_title("Effect of Temperature Decay Parameter for ArithDecay")
ax1.set_xlabel('Temperature Decay Parameter ')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(exp_consts, iteration_sa, label = 'Number of Iterations', color="C1", lw=2)
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.4), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig('OneMax_SA_Effect of Temperature Decay Parameter for ArithDecay.png',dpi=600)

#===============================================================================================
schedules = [mlrose.GeomDecay(init_temp =5,decay=0.9), mlrose.ArithDecay(init_temp =1,decay=0.005), mlrose.ExpDecay(init_temp =10,exp_const=0.015)]
fitness_curve_sas = []
best_state_sas = []
best_fitness_sas = []
iteration_sa = []
for i in range(len(schedules)):
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedules[i],
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state, curve=True, random_state = 1)
    best_state_sas.append(best_state_sa)
    best_fitness_sas.append(best_fitness_sa)
    fitness_curve_sas.append(fitness_curve_sa)

fig = plt.figure()

plt.plot(fitness_curve_sas[0], label = 'GeomDecay, init_temp =5, decay=0.9', color="C0", lw=2)
plt.plot(fitness_curve_sas[1], label = 'ArithDecay, init_temp =1, decay=0.005', color="C1", lw=2)
plt.plot(fitness_curve_sas[2], label = 'ExpDecay, init_temp =10, exp_const=0.015', color="C2", lw=2)

plt.ylabel('Fitness Value')
plt.xlabel('Iterations')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Fitness Value VS Iterations for Various Decay Schedules', y = 1.03)
plt.legend(loc=4)
#plt.ylim(0,40)
plt.grid(True)
plt.tight_layout()
plt.savefig('OneMax_SA_Fitness Value VS Iterations for Various Decay Schedules.png',dpi=600)

#================================================================================================
# Solve problem using genetic algorithm

population = [10,25,50,100,150,200,250,300,350,400,450,500]
fitness_curve_gas = []
best_state_gas = []
best_fitness_gas = []
iteration_ga = []
for i in range(len(population)):
    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem,
                                                      pop_size=population[i], mutation_prob=0.1,                                        
                                                      max_attempts = 100, max_iters = 1000,
                                                      curve=True, random_state = 1)
    best_state_gas.append(best_state_ga)
    best_fitness_gas.append(best_fitness_ga)
    fitness_curve_gas.append(fitness_curve_ga)
    iteration_ga.append(len(fitness_curve_ga))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(population, best_fitness_gas, label = 'Fitness Value', color="C0", lw=2)
ax1.set_ylabel('Fitness Value')
ax1.set_title("Effect of Size of Population for Genetic Algorithm")
ax1.set_xlabel('Size of Population')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(population, iteration_ga, label = 'Number of Iterations', color="C1", lw=2)
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(0.5,0.2), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig('OneMax_GA_Effect of Size of Population for Genetic Algorithm.png',dpi=600)

#=========================================================================================
mutation = [0.01, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
fitness_curve_gas = []
best_state_gas = []
best_fitness_gas = []
iteration_ga = []
for i in range(len(mutation)):
    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem,
                                                      pop_size=250, mutation_prob=mutation[i],                                        
                                                      max_attempts = 100, max_iters = 1000,
                                                      curve=True, random_state = 1)
    best_state_gas.append(best_state_ga)
    best_fitness_gas.append(best_fitness_ga)
    fitness_curve_gas.append(fitness_curve_ga)
    iteration_ga.append(len(fitness_curve_ga))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(mutation, best_fitness_gas, label = 'Fitness Value', color="C0", lw=2)
ax1.set_ylabel('Fitness Value')
ax1.set_title("Effect of Probability of Mutation for Genetic Algorithm")
ax1.set_xlabel('Probability of Mutation')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(mutation, iteration_ga, label = 'Number of Iterations', color="C1", lw=2)
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig('OneMax_GA_Effect of Probability of Mutation for Genetic Algorithm.png',dpi=600)

#================================================================================================
# Solve problem using MIMIC

population = [50,100,150,200,300,500]
fitness_curve_mimics = []
best_state_mimics = []
best_fitness_mimics = []
iteration_mimic = []
for i in range(len(population)):
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem,
                                                      pop_size=population[i], keep_pct=0.2,                                        
                                                      max_attempts = 10, max_iters = 1000,
                                                      curve=True, random_state = 1)
    best_state_mimics.append(best_state_mimic)
    best_fitness_mimics.append(best_fitness_mimic)
    fitness_curve_mimics.append(fitness_curve_mimic)
    iteration_mimic.append(len(fitness_curve_mimic))

fig = plt.figure()
plt.plot(population, best_fitness_mimics, label = 'Size of Population', color="navy", lw=2)
plt.ylabel('Fitness Value')
plt.xlabel('Size of Population')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Fitness Value VS Size of Population', y = 1.03)
#plt.legend()
#plt.ylim(0,40)
plt.grid(True)
plt.savefig('OneMax_MIMIC_Fitness Value VS Size of Population.png',dpi=600)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(population, best_fitness_mimics, label = 'Fitness Value', color="C0", lw=2)
ax1.set_ylabel('Fitness Value')
ax1.set_title("Effect of Size of Population for MIMIC")
ax1.set_xlabel('Size of Population')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(population, iteration_mimic, label = 'Number of Iterations', color="C1", lw=2)
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig('OneMax_MIMIC_Effect of Size of Population for MIMIC.png',dpi=600)

#==============================================================================
keep_pct = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
fitness_curve_mimics = []
best_state_mimics = []
best_fitness_mimics = []
iteration_mimic = []
for i in range(len(keep_pct)):
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem,
                                                      pop_size=200, keep_pct=keep_pct[i],                                        
                                                      max_attempts = 10, max_iters = 1000,
                                                      curve=True, random_state = 1)
    best_state_mimics.append(best_state_mimic)
    best_fitness_mimics.append(best_fitness_mimic)
    fitness_curve_mimics.append(fitness_curve_mimic)
    iteration_mimic.append(len(fitness_curve_mimic))

fig = plt.figure()
plt.plot(keep_pct, best_fitness_mimics, label = 'Fitness Value', color="navy", lw=2)
#plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], best_fitness_mimics, label = 'Proportion of Samples to Keep', color="navy", lw=2)
plt.ylabel('Fitness Value')
plt.xlabel('Proportion of Samples to Keep')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Fitness Value VS Proportion of Samples to Keep', y = 1.03)
#plt.legend()
#plt.ylim(0,40)
plt.grid(True)
plt.tight_layout()
plt.savefig('OneMax_MIMIC_Fitness Value VS Proportion of Samples to Keep.png',dpi=600)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(keep_pct, best_fitness_mimics, label = 'Fitness Value', color="C0", lw=2)
ax1.set_ylabel('Fitness Value')
ax1.set_title("Effect of Proportion of Samples to Keep for MIMIC")
ax1.set_xlabel('Proportion of Samples to Keep')
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(keep_pct, iteration_mimic, label = 'Number of Iterations', color="C1", lw=2)
ax2.set_ylabel('Number of Iterations')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig('OneMax_MIMIC_Effect of Proportion of Samples to Keep for MIMIC.png',dpi=600)


fig = plt.figure()
for i in range(len(best_fitness_mimics)):
    s = 'keep_pct = ' + str(keep_pct[i])
    color = 'C' + str(i)
    plt.plot(fitness_curve_mimics[i], label = s, color=color, lw=2)

plt.ylabel('Fitness Value')
plt.xlabel('Iterations')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Fitness Value VS Iterations for Various Keep Percentage', y = 1.03)
plt.legend()
#plt.ylim(0,40)
plt.grid(True)
plt.tight_layout()
plt.savefig('OneMax_MIMIC_Fitness Value VS Iterations for Various Keep Percentage.png',dpi=600)


#=======================================================================================
start_rhc = time.time()
best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem,
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state, restarts=2, curve=True, random_state = 1)
end_rhc = time.time()

print('#'*50)
print('random hill climb')
print('The best state found is: ', best_state_rhc)
print('The fitness at the best state is: ', best_fitness_rhc)
print('Optimization time is: ',end_rhc - start_rhc)
print('Number of effective iteration is: ',len(fitness_curve_rhc)-100)
print('Number of evaluation is: ',len(fitness_curve_rhc)-100)

# Solve problem using simulated annealing
# Define decay schedule
#schedule = mlrose.GeomDecay(decay = 0.99,init_temp=20)
schedule = mlrose.ExpDecay(exp_const  = 0.015,init_temp=10)

start_sa = time.time()
best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedule,
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state, curve=True, random_state = 1)
end_sa = time.time()

print('#'*50)
print('simulated annealing')
print('The best state found is: ', best_state_sa)
print('The fitness at the best state is: ', best_fitness_sa)
print('Optimization time is: ',end_sa - start_sa)
print('Number of effective iteration is: ',len(fitness_curve_sa)-100)
print('Number of evaluation is: ',len(fitness_curve_sa)-100)

# Solve problem using genetic algorithm
start_ga = time.time()
best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem,
                                                      pop_size=500, mutation_prob=0.001,                                        
                                                      max_attempts = 200, max_iters = 1000,
                                                      curve=True, random_state = 1)
end_ga = time.time()

print('#'*50)
print('genetic algorithm')
print('The best state found is: ', best_state_ga)
print('The fitness at the best state is: ', best_fitness_ga)
print('Optimization time is: ',end_ga - start_ga)
print('Number of effective iteration is: ',len(fitness_curve_ga)-200)
print('Number of evaluation is: ',500*(len(fitness_curve_ga)-200))

# Solve problem using MIMIC
start_mimic = time.time()
best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem,
                                                      pop_size=200, keep_pct=0.2,                                        
                                                      max_attempts = 10, max_iters = 1000,
                                                      curve=True, random_state = 1)
end_mimic = time.time()

print('#'*50)
print('MIMIC')
print('The best state found is: ', best_state_mimic)
print('The fitness at the best state is: ', best_fitness_mimic)
print('Optimization time is: ',end_mimic - start_mimic)
print('Number of effective iteration is: ',len(fitness_curve_mimic)-10)
print('Number of evaluation is: ',200*(len(fitness_curve_mimic)-10))


fig = plt.figure()
plt.plot(fitness_curve_rhc, label = 'Randomized Hill Climbing', color="C0", lw=2)
plt.plot(fitness_curve_sa, label = 'Simulated Annealing', color="C1", lw=2)
plt.plot(fitness_curve_ga, label = 'Genetic Algorithm', color="C2", lw=2)
plt.plot(fitness_curve_mimic, label = 'MIMIC', color="C3", lw=2)
plt.ylabel('Fitness Value')
plt.xlabel('Iterations')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Fitness Value VS Iterations for Random Optimization', y = 1.03)
plt.legend(loc=4)
plt.grid(True)
plt.tight_layout()
plt.savefig('OneMax_Fitness Value VS Iterations for Random Optimization.png',dpi=600)

#====================================================================================
# Define fitness function
sizes = [10,20,30,40,50,60,70,80,90,100]
fitness = mlrose.OneMax()

best_state_rhcs = []
fitness_curve_rhcs = []
best_fitness_mean_rhc = []
iteration_mean_rhc = []
evaluation_mean_rhc = []
best_fitness_std_rhc = []
iteration_std_rhc = []
evaluation_std_rhc = []
time_mean_rhc = []
time_std_rhc = []

best_state_sas = []
fitness_curve_sas = []
best_fitness_mean_sa = []
iteration_mean_sa = []
evaluation_mean_sa = []
best_fitness_std_sa = []
iteration_std_sa = []
evaluation_std_sa = []
time_mean_sa = []
time_std_sa = []

best_state_gas = []
fitness_curve_gas = []
best_fitness_mean_ga = []
iteration_mean_ga = []
evaluation_mean_ga = []
best_fitness_std_ga = []
iteration_std_ga = []
evaluation_std_ga = []
time_mean_ga = []
time_std_ga = []

best_state_mimics = []
fitness_curve_mimics = []
best_fitness_mean_mimic = []
iteration_mean_mimic = []
evaluation_mean_mimic = []
best_fitness_std_mimic = []
iteration_std_mimic = []
evaluation_std_mimic = []
time_mean_mimic = []
time_std_mimic = []

for i in range(len(sizes)):
    # Define problem object
    problem = mlrose.DiscreteOpt(length = sizes[i], fitness_fn = fitness,
                             maximize = True, max_val = 2)
    
    # Define initial state
    np.random.seed(seed=1)
    init_state = np.random.randint(0, 2, sizes[i])
    
    best_fitness_batch_rhc = []
    best_fitness_batch_sa = []
    best_fitness_batch_ga = []
    best_fitness_batch_mimic = []
    
    iteration_batch_rhc = []
    iteration_batch_sa = []
    iteration_batch_ga = []
    iteration_batch_mimic = []
    
    evaluation_batch_rhc = []
    evaluation_batch_sa = []
    evaluation_batch_ga = []
    evaluation_batch_mimic = []
    
    time_batch_rhc = []
    time_batch_sa = []
    time_batch_ga = []
    time_batch_mimic = []
    
    for k in range(5):
        start_rhc = time.time()
        best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, curve=True, restarts=50, random_state = k)
        end_rhc = time.time()
        best_fitness_batch_rhc.append(best_fitness_rhc)
        iteration_batch_rhc.append(len(fitness_curve_rhc))
        evaluation_batch_rhc.append(len(fitness_curve_rhc))
        time_batch_rhc.append(end_rhc-start_rhc)
        
        start_sa = time.time()
        best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, max_attempts=100, curve=True, random_state = k)
        end_sa = time.time()
        best_fitness_batch_sa.append(best_fitness_sa)
        iteration_batch_sa.append(len(fitness_curve_sa))
        evaluation_batch_sa.append(len(fitness_curve_sa))
        time_batch_sa.append(end_sa-start_sa)
        
        start_ga = time.time()
        best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, pop_size=200, max_attempts=10,curve=True, random_state = k)
        end_ga = time.time()
        best_fitness_batch_ga.append(best_fitness_ga)
        iteration_batch_ga.append(len(fitness_curve_ga))
        evaluation_batch_ga.append(len(fitness_curve_ga)*200)
        time_batch_ga.append(end_ga-start_ga)
        
        start_mimic = time.time()
        best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem, pop_size=50, curve=True, random_state = k)
        end_mimic = time.time()
        best_fitness_batch_mimic.append(best_fitness_mimic)
        iteration_batch_mimic.append(len(fitness_curve_mimic))
        evaluation_batch_mimic.append(len(fitness_curve_mimic)*50)
        time_batch_mimic.append(end_mimic-start_mimic)
        
    best_fitness_mean_rhc.append(np.mean(best_fitness_batch_rhc, axis=0))
    iteration_mean_rhc.append(np.mean(iteration_batch_rhc, axis=0))
    evaluation_mean_rhc.append(np.mean(evaluation_batch_rhc, axis=0))
    best_fitness_std_rhc.append(np.std(best_fitness_batch_rhc, axis=0))
    iteration_std_rhc.append(np.std(iteration_batch_rhc, axis=0))
    evaluation_std_rhc.append(np.std(evaluation_batch_rhc, axis=0))
    time_mean_rhc.append(np.mean(time_batch_rhc, axis=0))
    time_std_rhc.append(np.mean(time_std_rhc, axis=0))
    
    best_fitness_mean_sa.append(np.mean(best_fitness_batch_sa, axis=0))
    iteration_mean_sa.append(np.mean(iteration_batch_sa, axis=0))
    evaluation_mean_sa.append(np.mean(evaluation_batch_sa, axis=0))
    best_fitness_std_sa.append(np.std(best_fitness_batch_sa, axis=0))
    iteration_std_sa.append(np.std(iteration_batch_sa, axis=0))
    evaluation_std_sa.append(np.std(evaluation_batch_sa, axis=0))
    time_mean_sa.append(np.mean(time_batch_sa, axis=0))
    time_std_sa.append(np.mean(time_std_sa, axis=0))
    
    best_fitness_mean_ga.append(np.mean(best_fitness_batch_ga, axis=0))
    iteration_mean_ga.append(np.mean(iteration_batch_ga, axis=0))
    evaluation_mean_ga.append(np.mean(evaluation_batch_ga, axis=0))
    best_fitness_std_ga.append(np.std(best_fitness_batch_ga, axis=0))
    iteration_std_ga.append(np.std(iteration_batch_ga, axis=0))
    evaluation_std_ga.append(np.std(evaluation_batch_ga, axis=0))
    time_mean_ga.append(np.mean(time_batch_ga, axis=0))
    time_std_ga.append(np.mean(time_std_ga, axis=0))
    
    best_fitness_mean_mimic.append(np.mean(best_fitness_batch_mimic, axis=0))
    iteration_mean_mimic.append(np.mean(iteration_batch_mimic, axis=0))
    evaluation_mean_mimic.append(np.mean(evaluation_batch_mimic, axis=0))
    best_fitness_std_mimic.append(np.std(best_fitness_batch_mimic, axis=0))
    iteration_std_mimic.append(np.std(iteration_batch_mimic, axis=0))
    evaluation_std_mimic.append(np.std(evaluation_batch_mimic, axis=0))
    time_mean_mimic.append(np.mean(time_batch_mimic, axis=0))
    time_std_mimic.append(np.mean(time_std_mimic, axis=0))
    

best_fitness_mean_rhc = np.array(best_fitness_mean_rhc)
iteration_mean_rhc = np.array(iteration_mean_rhc)
evaluation_mean_rhc = np.array(evaluation_mean_rhc)
best_fitness_std_rhc = np.array(best_fitness_std_rhc)
iteration_std_rhc = np.array(iteration_std_rhc)
evaluation_std_rhc = np.array(evaluation_std_rhc)
time_mean_rhc = np.array(time_mean_rhc)
time_std_rhc = np.array(time_std_rhc)
    
best_fitness_mean_sa = np.array(best_fitness_mean_sa)
iteration_mean_sa = np.array(iteration_mean_sa)
evaluation_mean_sa = np.array(evaluation_mean_sa)
best_fitness_std_sa = np.array(best_fitness_std_sa)
iteration_std_sa = np.array(iteration_std_sa)
evaluation_std_sa = np.array(evaluation_std_sa)
time_mean_sa = np.array(time_mean_sa)
time_std_sa = np.array(time_std_sa)
    
best_fitness_mean_ga = np.array(best_fitness_mean_ga)
iteration_mean_ga = np.array(iteration_mean_ga)
evaluation_mean_ga = np.array(evaluation_mean_ga)
best_fitness_std_ga = np.array(best_fitness_std_ga)
iteration_std_ga = np.array(iteration_std_ga)
evaluation_std_ga = np.array(evaluation_std_ga)
time_mean_ga = np.array(time_mean_ga)
time_std_ga = np.array(time_std_ga)
    
best_fitness_mean_mimic = np.array(best_fitness_mean_mimic)
iteration_mean_mimic = np.array(iteration_mean_mimic)
evaluation_mean_mimic = np.array(evaluation_mean_mimic)
best_fitness_std_mimic = np.array(best_fitness_std_mimic)
iteration_std_mimic = np.array(iteration_std_mimic)
evaluation_std_mimic = np.array(evaluation_std_mimic)
time_mean_mimic = np.array(time_mean_mimic)
time_std_mimic = np.array(time_std_mimic)  
    
    
fig = plt.figure()
plt.plot(sizes,best_fitness_mean_rhc, label = 'Randomized Hill Climbing', color="C0", lw=2)
plt.fill_between(sizes, best_fitness_mean_rhc - best_fitness_std_rhc, best_fitness_mean_rhc + best_fitness_std_rhc, alpha=0.2,color="C0", lw=2)
plt.plot(sizes,best_fitness_mean_sa, label = 'Simulated Annealing', color="C1", lw=2)
plt.fill_between(sizes, best_fitness_mean_sa - best_fitness_std_sa, best_fitness_mean_sa + best_fitness_std_sa, alpha=0.2,color="C1", lw=2)
plt.plot(sizes,best_fitness_mean_ga, label = 'Genetic Algorithm', color="C2", lw=2)
plt.fill_between(sizes, best_fitness_mean_ga - best_fitness_std_ga, best_fitness_mean_ga + best_fitness_std_ga, alpha=0.2,color="C2", lw=2)
plt.plot(sizes,best_fitness_mean_mimic, label = 'MIMIC', color="C3", lw=2)
plt.fill_between(sizes, best_fitness_mean_mimic - best_fitness_std_mimic, best_fitness_mean_mimic + best_fitness_std_mimic, alpha=0.2,color="C3", lw=2)
plt.ylabel('Avergae Fitness Value', fontsize = 14)
plt.xlabel('Problem Size', fontsize = 14)
plt.title('Avergae Fitness Value VS Problem Size for OneMax Optimization', y = 1.03)
plt.legend(loc=0)
plt.grid(True)
plt.savefig('OneMax_Avergae Fitness Value VS Problem Size for OneMax Optimization.png',dpi=600)
#=================================================================================================
fig = plt.figure()
plt.plot(sizes,iteration_mean_rhc, label = 'Randomized Hill Climbing', color="C0", lw=2)
plt.fill_between(sizes, iteration_mean_rhc - iteration_std_rhc, iteration_mean_rhc + iteration_std_rhc, alpha=0.2,color="C0", lw=2)
plt.plot(sizes,iteration_mean_sa, label = 'Simulated Annealing', color="C1", lw=2)
plt.fill_between(sizes, iteration_mean_sa - iteration_std_sa, iteration_mean_sa + iteration_std_sa, alpha=0.2,color="C1", lw=2)
plt.plot(sizes,iteration_mean_ga, label = 'Genetic Algorithm', color="C2", lw=2)
plt.fill_between(sizes, iteration_mean_ga - iteration_std_ga, iteration_mean_ga + iteration_std_ga, alpha=0.2,color="C2", lw=2)
plt.plot(sizes,iteration_mean_mimic, label = 'MIMIC', color="C3", lw=2)
plt.fill_between(sizes, iteration_mean_mimic - iteration_std_mimic, iteration_mean_mimic + iteration_std_mimic, alpha=0.2,color="C3", lw=2)
plt.ylabel('Avergae Iterations', fontsize = 14)
plt.xlabel('Problem Size', fontsize = 14)
plt.title('Avergae Iterations Value VS Problem Size for OneMax Optimization', y = 1.03)
plt.legend(loc=0)
plt.grid(True)
plt.savefig('OneMax_Avergae Iterations Value VS Problem Size for OneMax Optimization.png',dpi=600)

#=================================================================================================
fig = plt.figure()
plt.plot(sizes,evaluation_mean_rhc, label = 'Randomized Hill Climbing', color="C0", lw=2)
plt.fill_between(sizes, evaluation_mean_rhc - evaluation_std_rhc, evaluation_mean_rhc + evaluation_std_rhc, alpha=0.2,color="C0", lw=2)
plt.plot(sizes,evaluation_mean_sa, label = 'Simulated Annealing', color="C1", lw=2)
plt.fill_between(sizes, evaluation_mean_sa - evaluation_std_sa, evaluation_mean_sa + evaluation_std_sa, alpha=0.2,color="C1", lw=2)
plt.plot(sizes,evaluation_mean_ga, label = 'Genetic Algorithm', color="C2", lw=2)
plt.fill_between(sizes, evaluation_mean_ga - evaluation_std_ga, evaluation_mean_ga + evaluation_std_ga, alpha=0.2,color="C2", lw=2)
plt.plot(sizes,evaluation_mean_mimic, label = 'MIMIC', color="C3", lw=2)
plt.fill_between(sizes, evaluation_mean_mimic - evaluation_std_mimic, evaluation_mean_mimic + evaluation_std_mimic, alpha=0.2,color="C3", lw=2)
plt.ylabel('Avergae Number of Evaluations', fontsize = 14)
plt.xlabel('Problem Size', fontsize = 14)
plt.title('Avergae Number of Evaluations VS Problem Size for OneMax Optimization', y = 1.03)
plt.legend(loc=0)
plt.grid(True)
plt.savefig('OneMax_Avergae Number of Evaluations VS Problem Size for OneMax Optimization.png',dpi=600)

#=================================================================================================
fig = plt.figure()
plt.plot(sizes,time_mean_rhc, label = 'Randomized Hill Climbing', color="C0", lw=2)
plt.fill_between(sizes, time_mean_rhc - time_std_rhc, time_mean_rhc + time_std_rhc, alpha=0.2,color="C0", lw=2)
plt.plot(sizes,time_mean_sa, label = 'Simulated Annealing', color="C1", lw=2)
plt.fill_between(sizes, time_mean_sa - time_std_sa, time_mean_sa + time_std_sa, alpha=0.2,color="C1", lw=2)
plt.plot(sizes,time_mean_ga, label = 'Genetic Algorithm', color="C2", lw=2)
plt.fill_between(sizes, time_mean_ga - time_std_ga, time_mean_ga + time_std_ga, alpha=0.2,color="C2", lw=2)
plt.plot(sizes,time_mean_mimic, label = 'MIMIC', color="C3", lw=2)
plt.fill_between(sizes, time_mean_mimic - time_std_mimic, time_mean_mimic + time_std_mimic, alpha=0.2,color="C3", lw=2)
plt.ylabel('Average Time', fontsize = 14)
plt.xlabel('Problem Size', fontsize = 14)
plt.title('Average Time VS Problem Size for OneMax Optimization', y = 1.03)
plt.legend(loc=0)
plt.grid(True)
plt.yscale('log')
plt.savefig('OneMax_Average Time VS Problem Size for OneMax Optimization.png',dpi=600)

#=====================================================================================
max_iters_rhc = [5,10,15,20,25,50,100,200,300,400,500,600,700,800,900,1000]
best_fitness_rhcs = []
iteration_rhc = []
for i in range(len(max_iters_rhc)):
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem,
                                                      max_attempts = 100, max_iters = max_iters_rhc[i],
                                                      init_state = init_state, restarts=2, curve=True, random_state = 1)
    best_fitness_rhcs.append(best_fitness_rhc)
    iteration_rhc.append(len(fitness_curve_rhc))

#=======================================================================================
max_iters_sa = [5,10,15,20,25,50,100,200,300,400,500,600,700,800,900,1000]
best_fitness_sas = []
iteration_sa = []

schedule = mlrose.ExpDecay(exp_const  = 0.015,init_temp=10)
for i in range(len(max_iters_sa)):
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedule,
                                                      max_attempts = 100, max_iters = max_iters_sa[i],
                                                      init_state = init_state, curve=True, random_state = 1)
    best_fitness_sas.append(best_fitness_sa)
    iteration_sa.append(len(fitness_curve_sa))

#===========================================================================================
max_iters_ga = [5,10,20,50,100,200,300,400,500,1000]
best_fitness_gas = []
iteration_ga = []
for i in range(len(max_iters_ga)):
    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem,
                                                      pop_size=500, mutation_prob=0.001,                                        
                                                      max_attempts = 200, max_iters = max_iters_ga[i],
                                                      curve=True, random_state = 1)
    best_fitness_gas.append(best_fitness_ga)
    iteration_ga.append(len(fitness_curve_ga))
#==========================================================================================
max_iters_mimic = [2,4,8,10,15,50,1000]
best_fitness_mimics = []
iteration_mimic = []
for i in range(len(max_iters_mimic)):
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem,
                                                      pop_size=200, keep_pct=0.2,                                        
                                                      max_attempts = 10, max_iters = max_iters_mimic[i],
                                                      curve=True, random_state = 1)
    best_fitness_mimics.append(best_fitness_mimic)
    iteration_mimic.append(len(fitness_curve_mimic))

fig = plt.figure()
plt.plot(max_iters_rhc, best_fitness_rhcs, label = 'Randomized Hill Climbing', color="C0", lw=2)
plt.plot(max_iters_sa, best_fitness_sas, label = 'Simulated Annealing', color="C1", lw=2)
plt.plot(max_iters_ga, best_fitness_gas, label = 'Genetic Algorithm', color="C2", lw=2)
plt.plot(max_iters_mimic, best_fitness_mimics, label = 'MIMIC', color="C3", lw=2)
plt.ylabel('Fitness Value')
plt.xlabel('Max Ieration')
#plt.xlabel('Training set size', fontsize = 14)
plt.title('Number of Iteration Required for Converging', y = 1.03)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('OneMax_Number of Iteration Required for Converging.png',dpi=600)
  

