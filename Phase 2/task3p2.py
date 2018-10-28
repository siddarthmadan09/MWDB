# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 20:45:31 2018

@author: vagarw14
"""

# -*- coding: utf-8 -*-
import operator
import csv
import sys
import numpy as np
import operator
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
# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse(r"/Users/sidmadan/Documents/mwdb/materials/DevSet/devset_topics.xml")
collection = DOMTree.documentElement
locationImageData = []     # A list to store all images data vectors for all locations
location_name_dict = {}    
imageIds=[]            # A list of all image IDS for all locations
imageScores = {}
simScoresLoc = {}
reducedLocations = []
AllRowsPerLoc =[0]
# CODE TO GET FILENAME OF ALL  LOCATION FILES
def getLocationNames():     
    topics = collection.getElementsByTagName("topic")
    for topic in topics:
        location_name_dict[int(topic.getElementsByTagName('number')[0].childNodes[0].data)] = topic.getElementsByTagName('title')[0].childNodes[0].data
    return location_name_dict

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
        dist = sum([(a - b)**2  for a, b in zip(inputLocVector, reducedLocations[idx])])**(1/2)
        simScoresLoc[location_name_dict[idx+1]] = dist
    return simScoresLoc               
    
def readInputLocation():
    loc_number= int(input('Enter location_number\n'))
    model = input('Enter the model of preference\n')
    decompositionMethod = input('Enter the Method of decomposition\n')
    model = model.upper()
    decompositionMethod = decompositionMethod.upper()
    kbest = int(input('K \n'))
    return loc_number,model,decompositionMethod,kbest

def readInputImg():
    img_id = int(input('Enter image id:\n'))
    model = input('Enter the model of preference\n')
    decompositionMethod = input('Enter the Method of decomposition\n')
    model = model.upper()
    decompositionMethod = decompositionMethod.upper()
    kbest = int(input('K \n'))
    return img_id,model,decompositionMethod,kbest 
    

def getImagedataByLoc(locationId, model,inputImageID,inputLocNumber):         
    arr=[]
    filenameX = location_name_dict[locationId] + ' ' + str(model)
    # print(filenameX)
    # CODE TO READ CSV LOCATION_MODEL FILE FOR LOCATION GIVEN
    with open("/Users/sidmadan/Documents/mwdb/materials/DevSet/descvis/img/"+filenameX+".csv","rt",newline='', encoding="utf8") as fp:
        
        line = fp.readline()
        while line:
            #arr = list(line.split(","))
            arr = line.split(",")
            arr[-1] = arr[-1][:-1]   # ALL IMAGE DATA WITHOUT IMAGE IDS
            arr= [round(float(x),3) for x in arr]
            if inputImageID==int(arr[0]):
                inputLocNumber = locationId
            imageIds.append(int(arr[0])) # ALL IMAGE IDS
            locationImageData.append(arr[1:])
            line = fp.readline()
#            locationAndImageId.append(locationX)
    AllRowsPerLoc.append(len(imageIds))
    return inputLocNumber
    
def getOneLocationVec(locationId, model,inputImageID,inputLocNumber):         
    arr=[]
    ImageData = []
    filenameX = location_name_dict[locationId] + ' ' + str(model)
    # print(filenameX)
    # CODE TO READ CSV LOCATION_MODEL FILE FOR LOCATION GIVEN
    with open("/Users/sidmadan/Documents/mwdb/materials/DevSet/descvis/img/"+filenameX+".csv","rt",newline='', encoding="utf8") as fp:
        
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
    Vt = Vt[:k,:]
    return np.dot(np_X, np.transpose(Vt))
                         
def computePca(X,k):
    np_X = np.asarray(X, dtype = float)
    covmatrix = np.cov(np.transpose(np_X))
    U, s, Vt = svd(covmatrix,full_matrices=False)
    Vt = Vt[:k,:]
    return np.dot(np_X, np.transpose(Vt))

  
#def lda_reduction(dataArray, k):
#    sparseDataArray = lil_matrix(dataArray)
#
#    model = lda.LDA(n_topics=20)
#    model.fit(sparseDataArray)  # model.fit_transform(X) is also available
#    topic_word = model.topic_word_  # model.components_ also works
#    n_top_words = k
#    for i, topic_dist in enumerate(topic_word):
#        topic_words = np.array(all_terms)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
#
#    doc_topic = model.doc_topic_
#    for i in range(10):
#        print("{} (top topic: {})".format(all_users[i], doc_topic[i].argmax()))

def lda_reduction(dataArray, k):
    sparseDataArray = lil_matrix(dataArray)

    model = lda.LDA(n_topics=k, n_iter=2)
    model.fit(sparseDataArray)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    doc_topic = model.doc_topic_
    latent = np.dot(dataArray,np.transpose(topic_word))
    return latent

def getLocationDataForLDA(np_locationImageData):
    shift = []
    for i in range(0,np_locationImageData.shape[1]):
        temp_arr = np_locationImageData[:,i]
        temp_arr = temp_arr[np.nonzero(temp_arr)]
        minx=100000
        for val in temp_arr:
            if float(val)> float(minx):
                val=minx
        np_locationImageData[:, i] = (np_locationImageData[:,i] + abs(val))/abs(val)
    np_locationImageData = np_locationImageData.astype(int)

    # print(np_locationImageData)
    latent = lda_reduction(np_locationImageData,4)
    print("LDA done")
    return latent

def main():
    # imgID,model,decompositionMethod,kGiven = readInputImg()
    imgID = 1288780397
    model = "CSD"
    decompositionMethod = "SVD"
    kGiven = "5"
#    readInputImg()
    startTime = datetime.now()
    inputLocNumber = -1
    location_name_dict = getLocationNames()
    
    for locationId in location_name_dict.keys() :
        inputLocNumber = getImagedataByLoc(locationId, model,imgID,inputLocNumber)
    
    print(np.array(locationImageData).shape)

   # lda_reduction(locationImageData,8)
    if decompositionMethod == "SVD":
        imageLatents = computeSvd(locationImageData, kGiven)
        print("SVD Latents ", np.array(imageLatents).shape)
        index = imageIds.index(imgID)
        inputImageVector = imageLatents[index]
        for row in range(0, np.size(imageLatents,0)):
            currentImageVector = imageLatents[row]
            currentImageID = imageIds[row]
            imageScores = findSimiliarityDist(inputImageVector, currentImageVector, currentImageID)

        imageScores  = sorted(imageScores.items(), key=operator.itemgetter(1))
        print(*imageScores[:5],sep = "\n")

        print("*" * 30)  # STARTING Comparison of Locations

        RowsPerLoc = getRowsperLoc()
        getLocationLatents(imageLatents,RowsPerLoc)
        inputLocVector =  getInputLocVector(reducedLocations,inputLocNumber)
        simScoresLoc = calculateSimscores(reducedLocations,inputLocVector)
        simScoresLoc  = sorted(simScoresLoc.items(), key=operator.itemgetter(1))
        print(*simScoresLoc[:5], sep = "\n")

    elif decompositionMethod == "PCA":
        imageLatents = computePca(locationImageData, kGiven)
        print("PCA Latests ", np.array(imageLatents).shape)
        index = imageIds.index(imgID)
        inputImageVector = imageLatents[index]
        for row in range(0, np.size(imageLatents,0)):
            currentImageVector = imageLatents[row]
            currentImageID = imageIds[row]
            imageScores = findSimiliarityDist(inputImageVector, currentImageVector, currentImageID)

        imageScores  = sorted(imageScores.items(), key=operator.itemgetter(1))
        print(*imageScores[:5],sep = "\n")

        print("*" * 30)  # STARTING Comparison of Locations

        RowsPerLoc = getRowsperLoc()
        getLocationLatents(imageLatents,RowsPerLoc)
        inputLocVector =  getInputLocVector(reducedLocations,inputLocNumber)
        simScoresLoc = calculateSimscores(reducedLocations,inputLocVector)
        simScoresLoc  = sorted(simScoresLoc.items(), key=operator.itemgetter(1))
        print(*simScoresLoc[:5], sep = "\n")
    elif decompositionMethod == "LDA":
        imageLatents = getLocationDataForLDA(np.array(locationImageData))
        index = imageIds.index(imgID)
        inputImageVector = imageLatents[index]
        for row in range(0, np.size(imageLatents,0)):
            currentImageVector = imageLatents[row]
            currentImageID = imageIds[row]
            imageScores = findSimiliarityDist(inputImageVector, currentImageVector, currentImageID)

        imageScores  = sorted(imageScores.items(), key=operator.itemgetter(1))
        print(*imageScores[:5],sep = "\n")

        print("*" * 30)  # STARTING Comparison of Locations

        RowsPerLoc = getRowsperLoc()
        getLocationLatents(imageLatents,RowsPerLoc)
        inputLocVector =  getInputLocVector(reducedLocations,inputLocNumber)
        simScoresLoc = calculateSimscores(reducedLocations,inputLocVector)
        simScoresLoc  = sorted(simScoresLoc.items(), key=operator.itemgetter(1))
        print(*simScoresLoc[:5], sep = "\n")
    else:
        print ("Invalid decomposition method")
        sys.exit(0) 
        
    print ("\nTotal time taken: ", datetime.now() - startTime)
        
main()
    
