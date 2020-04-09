############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Local Search-GRASP

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Local_Search-GRASP, File: Python-MH-Local Search-GRASP.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Local_Search-GRASP>

############################################################################

# Required Libraries
import pandas as pd
import random
import numpy  as np
import copy
import os
from matplotlib import pyplot as plt 

# Function: Tour Distance
def distance_calc(Xdata, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance = distance + Xdata[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: Euclidean Distance 
def euclidean_distance(x, y):       
    distance = 0
    for j in range(0, len(x)):
        distance = (x[j] - y[j])**2 + distance   
    return distance**(1/2) 

# Function: Initial Seed
def seed_function(Xdata):
    seed = [[],float("inf")]
    sequence = random.sample(list(range(1,Xdata.shape[0]+1)), Xdata.shape[0])
    sequence.append(sequence[0])
    seed[0] = sequence
    seed[1] = distance_calc(Xdata, seed)
    return seed

# Function: Build Distance Matrix
def buid_distance_matrix(coordinates):
    Xdata = np.zeros((coordinates.shape[0], coordinates.shape[0]))
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            if (i != j):
                x = coordinates[i,:]
                y = coordinates[j,:]
                Xdata[i,j] = euclidean_distance(x, y)        
    return Xdata

# Function: Tour Plot
def plot_tour_distance_matrix (Xdata, city_tour):
    m = np.copy(Xdata)
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            m[i,j] = (1/2)*(Xdata[0,j]**2 + Xdata[i,0]**2 - Xdata[i,j]**2)    
    w, u = np.linalg.eig(np.matmul(m.T, m))
    s = (np.diag(np.sort(w)[::-1]))**(1/2) 
    coordinates = np.matmul(u, s**(1/2))
    coordinates = coordinates.real[:,0:2]
    xy = np.zeros((len(city_tour[0]), 2))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy[:,0], xy[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy[0,0], xy[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy[1,0], xy[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: Tour Plot
def plot_tour_coordinates (coordinates, city_tour):
    xy = np.zeros((len(city_tour[0]), 2))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy[:,0], xy[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy[0,0], xy[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy[1,0], xy[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: Rank Cities by Distance
def ranking(Xdata, city = 0):
    rank = np.zeros((Xdata.shape[0], 2)) # ['Distance', 'City']
    for i in range(0, rank.shape[0]):
        rank[i,0] = Xdata[i,city]
        rank[i,1] = i + 1
    rank = rank[rank[:,0].argsort()]
    return rank

# Function: RCL
def restricted_candidate_list(Xdata, greediness_value = 0.5):
    seed = [[],float("inf")]
    sequence = []
    sequence.append(random.sample(list(range(1,Xdata.shape[0]+1)), 1)[0])
    count = 1
    for i in range(0, Xdata.shape[0]):
        count = 1
        rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        if (rand > greediness_value and len(sequence) < Xdata.shape[0]):
            next_city = int(ranking(Xdata, city = sequence[-1] - 1)[count,1])
            while next_city in sequence:
                count = np.clip(count+1,1,Xdata.shape[0]-1)
                next_city = int(ranking(Xdata, city = sequence[-1] - 1)[count,1])
            sequence.append(next_city)
        elif (rand <= greediness_value and len(sequence) < Xdata.shape[0]):
            next_city = random.sample(list(range(1,Xdata.shape[0]+1)), 1)[0]
            while next_city in sequence:
                next_city = int(random.sample(list(range(1,Xdata.shape[0]+1)), 1)[0])
            sequence.append(next_city)
    sequence.append(sequence[0])
    seed[0] = sequence
    seed[1] = distance_calc(Xdata, seed)
    return seed

# Function: 2_opt
def local_search_2_opt(Xdata, city_tour):
    tour = copy.deepcopy(city_tour)
    best_route = copy.deepcopy(tour)
    seed = copy.deepcopy(tour)  
    for i in range(0, len(tour[0]) - 2):
        for j in range(i+1, len(tour[0]) - 1):
            best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
            best_route[0][-1]  = best_route[0][0]                          
            best_route[1] = distance_calc(Xdata, best_route)           
            if (best_route[1] < tour[1]):
                tour[1] = copy.deepcopy(best_route[1])
                for n in range(0, len(tour[0])): 
                    tour[0][n] = best_route[0][n]          
            best_route = copy.deepcopy(seed) 
    return tour

def greedy_randomized_adaptive_search_procedure(Xdata, city_tour, iterations = 50, rcl = 25, greediness_value = 0.5):
    count = 0
    best_solution = copy.deepcopy(city_tour)
    while (count < iterations):
        rcl_list = []
        for i in range(0, rcl):
            rcl_list.append(restricted_candidate_list(Xdata, greediness_value = greediness_value))
        candidate = int(random.sample(list(range(0,rcl)), 1)[0])
        city_tour = local_search_2_opt(Xdata, city_tour = rcl_list[candidate])
        while (city_tour[0] != rcl_list[candidate][0]):
            rcl_list[candidate] = copy.deepcopy(city_tour)
            city_tour = local_search_2_opt(Xdata, city_tour = rcl_list[candidate])
        if (city_tour[1] < best_solution[1]):
            best_solution = copy.deepcopy(city_tour) 
        count = count + 1
    return best_solution
