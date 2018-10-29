import operator
import csv
import sys
import numpy as np
import operator
import statistics
from scipy.sparse import csr_matrix
from numpy.linalg import svd
from collections import defaultdict
import gensim,lda
from xml.dom.minidom import parse
import xml.dom.minidom
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from numpy import linalg
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import csv
import sys
from sys import argv

import os

# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse(r"./data/devset_topics.xml")
collection = DOMTree.documentElement
locationImageData = []     # A list to store all images data vectors for all locations
location_name_dict = {}    
imageIds=[]            # A list of all image IDS for all locations
imageScores = {}
simScoresLoc = {}
reducedLocations = []
representativeLocations = []
AllRowsPerLoc =[0]
# CODE TO GET FILENAME OF ALL  LOCATION FILES

# store the location id and names as key value
def getLocationNames():     
    topics = collection.getElementsByTagName("topic")
    for topic in topics:
        location_name_dict[int(topic.getElementsByTagName('number')[0].childNodes[0].data)] = topic.getElementsByTagName('title')[0].childNodes[0].data
    return location_name_dict

# compute similarity distance measure between given input img id and other images and store the scores for the compared img.
def findSimiliarityDist(inputImageVector, currentImageVector, currentImageID):
    dist = sum([(a - b)**2 for a, b in zip(inputImageVector, currentImageVector)])**(1/2)

    imageScores[int(currentImageID)] = dist
    return imageScores

def findSimiliarityDist1(inputVector, currentVector):
    dist = sum([(a - b)** 2 for a, b in zip(inputVector, currentVector)])**(1/2)
    return dist

def readInputs():
    loc_number= int(argv[1])
    model = argv[2]
    decompositionMethod = argv[3]
    model = model.upper()
    decompositionMethod = decompositionMethod.upper()
    kbest = int(argv[4])
    return loc_number,model,decompositionMethod,kbest

def getInputLocVector(X, inputLocNumber):
    inputLocVector = X[inputLocNumber-1]
    return inputLocVector

def calculateSimscores(reducedLocations,inputLocVector):

    for idx in range(0,len(reducedLocations)):
        dist = sum([(a - b)**2  for a, b in zip(inputLocVector, reducedLocations[idx])])**(1/2)
        simScoresLoc[location_name_dict[idx+1]] = dist
    return simScoresLoc
    
def findInputLocLatents(X,inputLocNumber):
    Y=[]
    for i in range(0,5):
        Y.append(X[i + (inputLocNumber-1)*5])
    return Y

# retrieve all the image data given location and color model
def getImagedataByLoc(locationId, model,inputImageID,inputLocNumber):
    arr=[]
    filenameX = location_name_dict[locationId] + ' ' + str(model)


    with open("./data/img/"+filenameX+".csv","rt", encoding="utf8") as fp:
        
        line = fp.readline()
        while line:
            arr = line.split(",")
            arr[-1] = arr[-1][:-1]   # ALL IMAGE DATA WITHOUT IMAGE IDS
            arr= [round(float(x),3) for x in arr]
            if inputImageID==int(arr[0]):
                inputLocNumber = locationId
            imageIds.append(int(arr[0])) # ALL IMAGE IDS
            locationImageData.append(arr[1:])
            line = fp.readline()
    AllRowsPerLoc.append(len(imageIds))
    return inputLocNumber

def populateLocationData(locationId, model):
    arr=[]
    filenameX = location_name_dict[locationId] + ' ' + str(model)
    locationX=[]
    # print(filenameX)
    # CODE TO READ CSV LOCATION_MODEL FILE FOR LOCATION GIVEN
    with open("./data/img/"+filenameX+".csv","rt", encoding="utf8") as fp:

        line = fp.readline()
        while line:
            #arr = list(line.split(","))
            arr = line.split(",")
            arr[-1] = arr[-1][:-1]   # ALL IMAGE DATA WITHOUT IMAGE IDS
            arr= [round(float(x),3) for x in arr]

            locationX.append(arr[1:])
            line = fp.readline()
        clusteredData = clusterData(locationX)
        for row in clusteredData:
            representativeLocations.append(row)


        return representativeLocations

# get the number of objects present in each location.
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
    Vt = Vt[:k,:]
    return np.dot(np_X, np.transpose(Vt))

def clusterData(X):
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    return kmeans.cluster_centers_

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
    imgID,model,decompositionMethod,kGiven = readInputs()

    startTime = datetime.now()
    inputLocNumber = -1

    location_name_dict = getLocationNames()
    
    for locationId in location_name_dict.keys() :
        inputLocNumber = getImagedataByLoc(locationId, model,imgID,inputLocNumber)

    if decompositionMethod.lower() == "svd":
        imageLatents = computeSvd(locationImageData, kGiven)

        print("LS matrix calulated")
        dataframe = pd.DataFrame(data=imageLatents.astype(float))
        dataframe.to_csv('outfile_'+sys.argv[3].upper()+"_"+str(datetime.now())+'.csv', sep=' ', header=False, float_format='%.2f', index=False)
        print("file save done")

        index = imageIds.index(imgID)
        inputImageVector = imageLatents[index]
        for row in range(0, np.size(imageLatents,0)):
            currentImageVector = imageLatents[row]
            currentImageID = imageIds[row]
            imageScores = findSimiliarityDist(inputImageVector, currentImageVector, currentImageID)

        imageScores  = sorted(imageScores.items(), key=operator.itemgetter(1))
        print(*imageScores[:5],sep = "\n")

        print("*" * 30)  # STARTING Comparison of Locations

        for locationId in location_name_dict.keys() :
            representativeLocations = populateLocationData(locationId, model)

        locationLatents = computePca(list(representativeLocations),kGiven)
        inputLocLatents = findInputLocLatents(locationLatents,inputLocNumber)

        ImgDist={}

        for loc_no in range(1,31):
            LocLatent = findInputLocLatents(locationLatents,loc_no)
            for i, v1 in enumerate(inputLocLatents):
                interClusDist =[]
                min_dist = float("inf")
                for j, v2 in enumerate(LocLatent):
                    min_dist = min(min_dist, findSimiliarityDist1(v1,v2))
                interClusDist.append(min_dist)
            ImgDist[location_name_dict[loc_no]] = statistics.mean(interClusDist)

        ImgDist  = sorted(ImgDist.items(), key=operator.itemgetter(1))
        print(*ImgDist[:5], sep = "\n")

    elif decompositionMethod.lower() == "pca":
        imageLatents = computePca(locationImageData, kGiven)

        print("LS matrix calulated")
        dataframe = pd.DataFrame(data=imageLatents.astype(float))
        dataframe.to_csv('outfile_'+sys.argv[3].upper()+"_"+str(datetime.now())+'.csv', sep=' ', header=False, float_format='%.2f', index=False)
        print("file save done")

        index = imageIds.index(imgID)
        inputImageVector = imageLatents[index]
        for row in range(0, np.size(imageLatents,0)):
            currentImageVector = imageLatents[row]
            currentImageID = imageIds[row]
            imageScores = findSimiliarityDist(inputImageVector, currentImageVector, currentImageID)

        imageScores  = sorted(imageScores.items(), key=operator.itemgetter(1))
        print(*imageScores[:5],sep = "\n")

        print("*" * 30)  # STARTING Comparison of Locations

        for locationId in location_name_dict.keys() :
            representativeLocations = populateLocationData(locationId, model)

        locationLatents = computePca(list(representativeLocations),kGiven)
        inputLocLatents = findInputLocLatents(locationLatents,inputLocNumber)

        ImgDist={}

        for loc_no in range(1,31):
            LocLatent = findInputLocLatents(locationLatents,loc_no)
            for i, v1 in enumerate(inputLocLatents):
                interClusDist =[]
                min_dist = float("inf")
                for j, v2 in enumerate(LocLatent):
                    min_dist = min(min_dist, findSimiliarityDist1(v1,v2))
                interClusDist.append(min_dist)
            ImgDist[location_name_dict[loc_no]] = statistics.mean(interClusDist)

        ImgDist  = sorted(ImgDist.items(), key=operator.itemgetter(1))
        print(*ImgDist[:5], sep = "\n")

    elif decompositionMethod.lower() == "lda":
        imageLatents = getLocationDataForLDA(np.array(locationImageData) , kGiven)

        print("LS matrix calulated")
        dataframe = pd.DataFrame(data=imageLatents.astype(float))
        dataframe.to_csv('outfile_'+sys.argv[3].upper()+"_"+str(datetime.now())+'.csv', sep=' ', header=False, float_format='%.2f', index=False)
        print("file save done")

        index = imageIds.index(imgID)
        inputImageVector = imageLatents[index]
        for row in range(0, np.size(imageLatents,0)):
            currentImageVector = imageLatents[row]
            currentImageID = imageIds[row]
            imageScores = findSimiliarityDist(inputImageVector, currentImageVector, currentImageID)

        imageScores  = sorted(imageScores.items(), key=operator.itemgetter(1))
        print(*imageScores[:5],sep = "\n")

        print("*" * 30)  # STARTING Comparison of Locations

        for locationId in location_name_dict.keys() :
            representativeLocations = populateLocationData(locationId, model)

        locationLatents = computePca(list(representativeLocations),kGiven)
        inputLocLatents = findInputLocLatents(locationLatents,inputLocNumber)

        ImgDist={}

        for loc_no in range(1,31):
            LocLatent = findInputLocLatents(locationLatents,loc_no)
            for i, v1 in enumerate(inputLocLatents):
                interClusDist =[]
                min_dist = float("inf")
                for j, v2 in enumerate(LocLatent):
                    min_dist = min(min_dist, findSimiliarityDist1(v1,v2))
                interClusDist.append(min_dist)
            ImgDist[location_name_dict[loc_no]] = statistics.mean(interClusDist)

        ImgDist  = sorted(ImgDist.items(), key=operator.itemgetter(1))
        print(*ImgDist[:5], sep = "\n")
    else:
        print ("Invalid decomposition method")
        sys.exit(0) 
        
    print ("\nTotal time taken: ", datetime.now() - startTime)
        
main()
    
