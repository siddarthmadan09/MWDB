# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:19:53 2018

@author: vagarw14
"""
# -*- coding: utf-8 -*-

import csv
import sys
import numpy as np
import operator
from scipy.sparse import csr_matrix
from numpy.linalg import svd
from collections import defaultdict
from xml.dom.minidom import parse
import xml.dom.minidom
import pandas as pd
from sklearn.cluster import KMeans
# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse(r"C:/Users/vagarw14/mwdb3/devset_topics.xml")
collection = DOMTree.documentElement
locationImageData = []     # A list to store all images data vectors for all locations
location_name_dict = {}    
imageIds=[]                    # A list of all image IDS for all locations
imageScores = {}
representativeLocations=[]
simScoresLoc = {}

AllRowsPerLoc =[0]
# CODE TO GET FILENAME OF ALL  LOCATION FILES
def getLocationNames():     
    topics = collection.getElementsByTagName("topic")
    for topic in topics:
        location_name_dict[int(topic.getElementsByTagName('number')[0].childNodes[0].data)] = topic.getElementsByTagName('title')[0].childNodes[0].data
    return location_name_dict


# print(location_name_dict)

# RIGHT NOW LOCATION ID IS CONVERTED TO INT inside dictionary
def findSimiliarityDist(inputImageVector, currentImageVector, currentImageID):
    dist = sum([(a - b)** 2 for a, b in zip(inputImageVector, currentImageVector)])**(1/2)
#    imageScores.setdefault(currentImageID,[]).append(dist)
    imageScores[int(currentImageID)] = dist
    return imageScores

def getInputLocVector(X, inputLocNumber):
    inputLocVector = X[inputLocNumber-1]
    return inputLocVector

def calculateSimscores(reducedLocations,inputLocVector):
#    inputLocVector = reducedLocations[inputLocNumber-1]
    
    for idx in range(0,len(reducedLocations)):
        dist = sum([abs(a - b)  for a, b in zip(inputLocVector, reducedLocations[idx])])
        simScoresLoc[location_name_dict[idx+1]] = dist
    return simScoresLoc               
    
def readInput():
    loc_number= int(input('Enter location_number\n'))
    model = input('Enter the model of preference\n')
    kbest= int(input('K \n'))

def readInputImg():
    img_id = int(input('Enter image id:\n'))
    model = input('Enter the model of preference\n')
    kbest = int(input('K \n'))
    
def clusterData(X):
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    return kmeans.cluster_centers_

def getImagedataByLoc(locationId, model):         
    arr=[]
    filenameX = location_name_dict[locationId] + ' ' + str(model)
    locationX=[]
    # print(filenameX)
    # CODE TO READ CSV LOCATION_MODEL FILE FOR LOCATION GIVEN
    with open("C:/Users/vagarw14/mwdb3/descvis/descvis/img/"+filenameX+".csv","rt",newline='', encoding="utf8") as fp:
        
        line = fp.readline()
        while line:
            #arr = list(line.split(","))
            arr = line.split(",")
            arr[-1] = arr[-1][:-1]   # ALL IMAGE DATA WITHOUT IMAGE IDS
            arr= [round(float(x),3) for x in arr]

            locationX.append(arr[1:])
            line = fp.readline()
    clusteredData = clusterData(locationX)        
    reduced_locationX = [sum(x)/len(clusteredData) for x in zip(*clusteredData)]                       
    representativeLocations.append(reduced_locationX)
    
    return representativeLocations
    
def getOneLocationVec(locationId, model,inputImageID,inputLocNumber):         
    arr=[]
    ImageData = []
    filenameX = location_name_dict[locationId] + ' ' + str(model)
    # print(filenameX)
    # CODE TO READ CSV LOCATION_MODEL FILE FOR LOCATION GIVEN
    with open("C:/Users/vagarw14/mwdb3/descvis/descvis/img/"+filenameX+".csv","rt",newline='', encoding="utf8") as fp:
        
        line = fp.readline()
        while line:
            #arr = list(line.split(","))
            arr = line.split(",")
            arr[-1] = arr[-1][:-1]   # ALL IMAGE DATA WITHOUT IMAGE IDS
            arr= [round(float(x),3) for x in arr]
            if inputImageID==int(arr[0]):
                inputLocNumber = locationId
            ImageData.append(arr[1:])
            line = fp.readline()

        reducedLocationX = [sum(x)/len(ImageData) for x in zip(*ImageData)]
        reducedLocations.append(reducedLocationX)
    return inputLocNumber
#            locationAndImageId.append(locationX)
def getRowsperLoc():
    RowsPerLoc=[]
    for i in range(1, len(AllRowsPerLoc)):
        RowsPerLoc.append(AllRowsPerLoc[i]-AllRowsPerLoc[i-1])

    return RowsPerLoc
        
def getLocationLatents(imageLatents,RowsPerLoc):
    start=0
    for rows in RowsPerLoc:
        ImageData = imageLatents[start:start+rows]
        reducedLocationX = [sum(x)/len(ImageData) for x in zip(*ImageData)]
        reducedLocations.append(reducedLocationX)
        start = rows
        
def computeSvd(X,k):
    np_X = np.asarray(X, dtype = float) # CONVERTING locationImageData to np_locationImageData
    U, s, Vt = svd(X,full_matrices=False)
#    print("U")
#    U = U [:,:k]
#    print(np.array(U).shape)
#    print("*" * 30)
#    print(s)
#    print("*" * 30)
    Vt = Vt[:k,:]
#    print(np.array(Vt).shape)
    return np.dot(np_X, np.transpose(Vt))
                         
def computePca(X,k):
    np_X = np.asarray(X, dtype = float)
    covmatrix = np.cov(np.transpose(np_X))
#    print(covmatrix.shape)
    U, s, Vt = svd(covmatrix,full_matrices=False)
##    print("U")
##    U = U [:,:k]
#    print(np.array(U).shape)
#    print("*" * 30)
#    print(s)
#    print("*" * 30)
    Vt = Vt[:k,:]
#    print(np.array(Vt).shape)
    return np.dot(np_X, np.transpose(Vt))
    
def main():
#    readInput()
#    readInputImg()
    inputLocData = []
    inputLocNumber = 6
    location_name_dict = getLocationNames()
    for locationId in location_name_dict.keys() :
        representativeLocations = getImagedataByLoc(locationId, "CM")
    print(np.array(representativeLocations).shape)
    
#    locationLatents = computeSvd(representativeLocations,5);
    locationLatents = computePca(representativeLocations,5);                            
    inputLocVector =  getInputLocVector(locationLatents,inputLocNumber)
    simScoresLoc = calculateSimscores(locationLatents,inputLocVector)
    simScoresLoc  = sorted(simScoresLoc.items(), key=operator.itemgetter(1))
    print(*simScoresLoc[:5], sep = "\n")
#    for i in range(0,4):
#        inputLocData.append(representativeLocations[i + (inputLocNumber-1)*4])
        
        
    
    
#    print(len(representativeLocations[2][1])
    
#    for(locdataX in representativeLocations)    
#    print(inputLocNumber, location_name_dict[inputLocNumber])
    
#    imageLatents = computeSvd(5)
#    print("Image Latests ", np.array(imageLatents).shape)
#    pcaLatents = computePca(5)
#    print("PCA Latests ",np.array(pcaLatents).shape)
#    imageLatents = pcaLatents
#    index = imageIds.index(10686632825)
#    inputImageVector = imageLatents[index]
#    for row in range(0, np.size(imageLatents,0)):
#        currentImageVector = imageLatents[row]
#        currentImageID = imageIds[row]
#        imageScores = findSimiliarityDist(inputImageVector, currentImageVector, currentImageID)    
#    
#    imageScores  = sorted(imageScores.items(), key=operator.itemgetter(1))    
#    print(*imageScores[:5],sep = "\n")
#    
#    print("*" * 30)  # STARTING Comparison of Locations
#    
#    RowsPerLoc = getRowsperLoc()
##    print((RowsPerLoc))
#    getLocationLatents(imageLatents,RowsPerLoc)
##    print(len(reducedLocations))
#    simScoresLoc = calculateSimscores(reducedLocations,inputLocNumber)
#    simScoresLoc  = sorted(simScoresLoc.items(), key=operator.itemgetter(1))
#    print(*simScoresLoc[:5], sep = "\n")
#    kmeans = KMeans(n_clusters=8, random_state=0).fit(imageLatents)
#    print(kmeans.cluster_centers_)
##    print(kmeans.labels_)
#    centroids = KMeans.fit_predict(X, y=None, sample_weight=None)[source]
#    print(centroids)
main()
    
    
