# -*- coding: utf-8 -*-

import csv
import sys
import statistics
import numpy as np
import operator
from scipy.sparse import csr_matrix
from numpy.linalg import svd
from collections import defaultdict
from xml.dom.minidom import parse
from scipy.sparse import lil_matrix
import gensim,lda
import xml.dom.minidom
import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import sys
from sys import argv

# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse(r"./data/devset_topics.xml")
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

def findSimiliarityDist(inputVector, currentVector):
    dist = sum([(a - b)** 2 for a, b in zip(inputVector, currentVector)])**(1/2)
    return dist

def getInputLocVector(X, inputLocNumber):
    inputLocVector = X[inputLocNumber-1]
    return inputLocVector

def calculateSimscores(reducedLocations,inputLocVector):
#    inputLocVector = reducedLocations[inputLocNumber-1]
    
    for idx in range(0,len(reducedLocations)):
        dist = sum([abs(a - b)  for a, b in zip(inputLocVector, reducedLocations[idx])])
        simScoresLoc[location_name_dict[idx+1]] = dist
    return simScoresLoc               
    
def readInputLocation():
    loc_number= int(argv[1])
    model = argv[2]
    decompositionMethod = argv[3]
    model = model.upper()
    decompositionMethod = decompositionMethod.upper()
    kbest = int(argv[4])
    return loc_number,model,decompositionMethod,kbest

def readInputImg():
    img_id = int(input('Enter image id:\n'))
    model = input('Enter the model of preference\n')
    kbest = int(input('K \n'))
    
def clusterData(X):
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    return kmeans.cluster_centers_

def getImagedataByLoc(locationId, model, cluster):
    arr=[]
    filenameX = location_name_dict[locationId] + ' ' + str(model)
    locationX=[]
    # print(filenameX)
    # CODE TO READ CSV LOCATION_MODEL FILE FOR LOCATION GIVEN
    with open("./data/img/"+filenameX+".csv","rt",newline='', encoding="utf8") as fp:
        
        line = fp.readline()
        while line:
            #arr = list(line.split(","))
            arr = line.split(",")
            arr[-1] = arr[-1][:-1]   # ALL IMAGE DATA WITHOUT IMAGE IDS
            arr= [round(float(x),3) for x in arr]

            locationX.append(arr[1:])
            line = fp.readline()
    if(bool(cluster)):
        clusteredData = clusterData(locationX)
        for row in clusteredData:
            representativeLocations.append(row)

    if(bool(cluster)):
        return representativeLocations
    else:
        return locationX
        
def computeSvd(X,k):
    np_X = np.asarray(X, dtype = float) # CONVERTING locationImageData to np_locationImageData
    U, s, Vt = svd(X,full_matrices=False)
    Vt = Vt[:k,:]
    return np.dot(np_X, np.transpose(Vt))
                         
def computePca(X,k):
    np_X = np.asarray(X, dtype = float)
    covmatrix = np.cov(np.transpose(np_X))
    U, s, Vt = svd(covmatrix,full_matrices=False)
    Vt = Vt[:k,:]
    return np.dot(np_X, np.transpose(Vt))

def lda_reduction(dataArray, k):
    sparseDataArray = lil_matrix(dataArray)

    model = lda.LDA(n_topics=k, n_iter=2)
    model.fit(sparseDataArray)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    doc_topic = model.doc_topic_
    latent = np.dot(dataArray,np.transpose(topic_word))
    return latent

def findInputLocLatents(X,inputLocNumber):
    Y=[]
    for i in range(0,5):
        Y.append(X[i + (inputLocNumber-1)*5])
    return Y

def getLocationDataForLDA(locationImageValues , k):
    np_locationImageValues = np.asarray(locationImageValues)
    scaler = MinMaxScaler()
    scaler.fit(np_locationImageValues)
    np_locationImageValues = scaler.transform(np_locationImageValues) * 1000

    rows, columns = np_locationImageValues.shape

    for i in range(rows):
        for j in range(columns):
            np_locationImageValues[i][j] = round(np_locationImageValues[i][j])

    np_locationImageValues = np_locationImageValues.astype(int)
    return lda_reduction(np_locationImageValues, k)

def main():

    inputLocNumber, model, decompositionMethod, kbest = readInputLocation()

    startTime = datetime.now()
    inputLocLatents = []
    location_name_dict = getLocationNames()
    representativeLocation = getImagedataByLoc(3, model, cluster=False)
    for locationId in location_name_dict.keys() :
        representativeLocations = getImagedataByLoc(locationId, model, cluster=True)


    if decompositionMethod.lower() == "svd":
        locationLatents = computeSvd(list(representativeLocation),kbest);

        print("LS matrix calulated")
        dataframe = pd.DataFrame(data=locationLatents.astype(float))
        dataframe.to_csv('outfile_'+sys.argv[3].upper()+"_"+str(datetime.now())+'.csv', sep=' ', header=False, float_format='%.2f', index=False)
        print("file save done")
        locationLatents = computeSvd(list(representativeLocations),kbest);

        inputLocLatents = findInputLocLatents(locationLatents,inputLocNumber)


        ImgDist={}

        for loc_no in range(1,31):
            LocLatent = findInputLocLatents(locationLatents,loc_no)
            for i, v1 in enumerate(inputLocLatents):
                interClusDist =[]
                min_dist = float("inf")
                for j, v2 in enumerate(LocLatent):
                    min_dist = min(min_dist, findSimiliarityDist(v1,v2))
                interClusDist.append(min_dist)
            ImgDist[location_name_dict[loc_no]] = statistics.mean(interClusDist)

        ImgDist  = sorted(ImgDist.items(), key=operator.itemgetter(1))
        print(*ImgDist[:5], sep = "\n")
        
    elif decompositionMethod.lower() == "pca":
        locationLatents = computePca(list(representativeLocation),kbest);

        print("LS matrix calulated")
        dataframe = pd.DataFrame(data=locationLatents.astype(float))
        dataframe.to_csv('outfile_'+sys.argv[3].upper()+"_"+str(datetime.now())+'.csv', sep=' ', header=False, float_format='%.2f', index=False)
        print("file save done")

        locationLatents = computePca(list(representativeLocations),kbest);
        inputLocLatents = findInputLocLatents(locationLatents,inputLocNumber)
       
        ImgDist={}
        
        for loc_no in range(1,31):
            LocLatent = findInputLocLatents(locationLatents,loc_no)
            for i, v1 in enumerate(inputLocLatents):
                interClusDist =[]
                min_dist = float("inf")
                for j, v2 in enumerate(LocLatent):
                    min_dist = min(min_dist, findSimiliarityDist(v1,v2))
                interClusDist.append(min_dist)                    
            ImgDist[location_name_dict[loc_no]] = statistics.mean(interClusDist)
    
        ImgDist  = sorted(ImgDist.items(), key=operator.itemgetter(1))            
        print(*ImgDist[:5], sep = "\n")

    elif decompositionMethod.lower() == "lda":
        locationLatents = getLocationDataForLDA(representativeLocation,kbest);

        print("LS matrix calulated")
        dataframe = pd.DataFrame(data=locationLatents.astype(float))
        dataframe.to_csv('outfile_'+sys.argv[3].upper()+"_"+str(datetime.now())+'.csv', sep=' ', header=False, float_format='%.2f', index=False)
        print("file save done")

        locationLatents = getLocationDataForLDA(representativeLocations,kbest);
        inputLocLatents = findInputLocLatents(locationLatents,inputLocNumber)

        ImgDist={}

        for loc_no in range(1,31):
            LocLatent = findInputLocLatents(locationLatents,loc_no)
            for i, v1 in enumerate(inputLocLatents):
                interClusDist =[]
                min_dist = float("inf")
                for j, v2 in enumerate(LocLatent):
                    min_dist = min(min_dist, findSimiliarityDist(v1,v2))
                interClusDist.append(min_dist)
            ImgDist[location_name_dict[loc_no]] = statistics.mean(interClusDist)

        ImgDist  = sorted(ImgDist.items(), key=operator.itemgetter(1))
        print(*ImgDist[:5], sep = "\n")
    else:
        print ("Invalid decomposition method")
        sys.exit(0) 
                
        
    print ("\nTotal time taken: ", datetime.now() - startTime)

main()
    
    
