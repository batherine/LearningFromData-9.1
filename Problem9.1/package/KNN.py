'''
Created on Apr 6, 2017

@author: catherine
'''
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from operator import itemgetter
import sympy.plotting.plot as plot
from pip._vendor.distlib.compat import raw_input

#a function i now know we do not need
def generatePoints(testSet, N):
    for i in range(N) :
        x1 = random.uniform(0,1) + random.randint(0.0,20.0) - 11.0
        x2 = random.uniform(0,1) + random.randint(0.0,20.0) - 11.0
        testSet.append([x1,x2, 0])

#calculate distance between two points        
def euclideanDistance(instance1, instance2,length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
    
#find neighbors for for single test vector   
#takes 1 test vector, and compare to all training set vectors 
def getNeighbors(vec1, dataSet):
    distances = []   
    for j in range(len(dataSet)) :
        vec2 = dataSet[j]
        vec2 = vec2.tolist()
        distance = euclideanDistance(vec1, vec2, 2)
        if(distance != 0) :
            distances.append([distance, vec2])        
    distances.sort(key=itemgetter(0))  
    return distances

#take label of closest neighbor and label test point with same label
def label(vec, neighbors):
    closestDist = neighbors[0]
    closestNeighbor = closestDist[1]
    vec[2] = closestNeighbor[2]
    
    
#plots the points
def plot(trainingSet):
    for i in range(len(trainingSet)) :
        if len(trainingSet[i]) > 2 :
            current = trainingSet[i]
            if(current[2] <0):
                plt.plot(current[0],current[1],'xr')
            else :
                plt.plot(current[0],current[1],'ob')

#plots line distinguishing the decision regions    
def decisionRegions (dataSet):
    for i in range(len(dataSet)):
        x1 = dataSet[i]
        for j in range(len(dataSet)):
            x2 = dataSet[j]
            if(x1[2] != x2[2]) :
                midpoint = [(x1[0] +x2[0])/2 , (x1[1] +x2[1])/2]
                slope = np.reciprocal((x1[1]-x2[1])/ (x1[0]-x2[0]) *-1)
                yint = (slope*midpoint[0] - midpoint[1]) * -1
                
                x=np.linspace(-2, x2[0]+3)
                plt.plot(x,slope*x + yint , '--k')

#literally the code from the scipy.cluster.vq.whiten package
#but it wasn't importing so i just added dat boi                
def whiten(obs, check_finite=True):
    std_dev = np.std(obs, axis=0)
    zero_std_mask = std_dev == 0
    if zero_std_mask.any():
        std_dev[zero_std_mask] = 1.0
    return obs / std_dev

          
def main():
    # prepare data
    numPoints = 100
    trainingSet = np.array([[0,0,1],[0,1.0,1],[5.0,5.0,-1]])
    testSet=[]
    #generatePoints(testSet,numPoints)
    print("Training Set :")
    print(trainingSet)
    
    var = raw_input("Select problem : (a,b, or c)")
    if var == "a" :
        for i in range(len(trainingSet)):
            neighbors = getNeighbors(trainingSet[i], trainingSet)
            print("Closest Neighbors for :")
            print(trainingSet[i])
            print(neighbors)
            decisionRegions(trainingSet)
        plot(trainingSet)    
            
    if var == "b" : 
        whitened = whiten(trainingSet)
        print("Whitened :")
        print(whitened)
        for i in range(len(trainingSet)):
            neighbors = getNeighbors(whitened[i], whitened)
            print("Closest Neighbors for :")
            print(whitened[i])
            print(neighbors)
            decisionRegions(whitened)     
        plot(whitened)
        
    if var == "c":
        trainingSet = np.array([[0,0],[0,1.0],[5.0,5.0]])
        pca = matplotlib.mlab.PCA(trainingSet)
        p =np.array(pca.Y)
        p=np.hstack((p, np.atleast_2d([1,1,-1]).T))
        print("Principal Component Analysis :")
        print(p)
        for i in range(len(p)) :
            neighbors= getNeighbors(p[i], p)
            print("Closest Neighbors for :")
            print(p[i])
            print(neighbors)
            decisionRegions(p)
        plot(p)
    
    
    plt.grid(True)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.xlim(-2,5.5)
    plt.ylim(-2,5.5)
    plt.show()
    #print("Test Set:")
    #print(testSet)
    
    
main()