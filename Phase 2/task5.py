import csv
import sys
import math
import os
import xml.dom.minidom
import datetime
#from scipy.sparse import csr_matrix
#from sklearn.decomposition import TruncatedSVD
#from sklearn.random_projection import sparse_random_matrix
#import scipy.sparse
#from numpy.linalg import svd
import numpy as np
from numpy import dot
from numpy.linalg import norm
#from sklearn.decomposition import PCA
from numpy.linalg import svd
from scipy import spatial
#from sklearn.decomposition import LatentDirichletAllocation

startTime = datetime.datetime.now()
fileName = ''
locationName = ""
otherLocationNames=[]
targetFileNames = []
fileNamesToCompare = []

def getCSVDataAsListData(fileName):
    mainData = []
    with open(fileName) as csv_file:
        csvData = csv.reader(csv_file, delimiter=',')
        for row in csvData:
            mainData.append(row)
        return mainData

def reduceVectorsToSingleVector(fileData):
    finalVector=[]
    for colIndex in range(len(fileData[0])):
        rowIndex = 0
        sum = 0
        while rowIndex<len(fileData):
            sum+=fileData[rowIndex][colIndex]
            rowIndex+=1
        finalVector.append(sum/len(fileData[0]))
    return finalVector

def returnSingleReducedVectorFOrAllModelsForLocation(targetFileNames):
    finalVector = []
    for idx, file in enumerate(targetFileNames):
        #if idx == 0:
            print "File Being compared = " + file
            data = getCSVDataAsListData(file)
            originalDataMatrix = []
            for row in data:
                tempArr = []
                for index, col in enumerate(row):
                    if index != 0:
                        tempArr.append((float)(col))
                originalDataMatrix.append(tempArr)

            print "Original Data Matrix Ready for " + str(file)
            print "Rows = " + str(len(originalDataMatrix)) + "Cols = " + str(len(originalDataMatrix[0]))
            if sys.argv[3].upper() == "SVD":
                U, singularValues, V = svd(originalDataMatrix, full_matrices=False)
            elif sys.argv[3].upper() == "PCA":
                print "PCA partially implemented"
                covMatrix = np.cov(originalDataMatrix)
                U, singularValues, V = svd(covMatrix, full_matrices=False)
            elif sys.argv[3].upper() == "LDA":
                print "LDA not implemented"
                sys.exit()
            else:
                print "Incorrect Model Entered.. Try Again"
                sys.exit()
            # print "Core matrix"
            # print singularValues
            # print "Rows = " + str(len(singularValues))
            # reverse_order = np.sort(singularValues)[::-1]
            # print "Rows after sorting= " + str(len(singularValues))
            # print singularValues

            # pick up first k values from
            reducedArr = []
            reducedArr = V[:(int)(sys.argv[2]), :]
            print "3rd matrix - Reduced"
            # print reducedArr
            print "Rows = " + str(len(reducedArr)) + " Out of " + str(len(V))
            print "Columns = " + str(len(reducedArr[0])) + " Out of " + str(len(V[0]))

            transArr = np.transpose(reducedArr)
            print "Transposed Reduced matrix created with rows = " + str(len(transArr)) + " and columns = " + str(len(transArr[0]))
            # get Object x LS matrix
            if sys.argv[3].upper() == "SVD":
                objectLSMatrix = np.dot(originalDataMatrix, transArr)
            elif sys.argv[3].upper() == "PCA":
                objectLSMatrix = np.dot(covMatrix, transArr)
            print "Object x LS matrix created with rows = " + str(len(objectLSMatrix)) + " and columns = " + str(
                len(objectLSMatrix[0]))
            # reduce all objects to single object feature vector
            reducedVector = reduceVectorsToSingleVector(objectLSMatrix)
            print "File reduced to single vector of length = " + str(len(reducedVector))
            print reducedVector
            finalVector.append(reducedVector)
    return reduceVectorsToSingleVector(finalVector)

def calculateSimilarityScoreUsingL1(targetVector,vectorBeingCompared):
    score = 0.0
    for index in range(len(targetVector)):
        score+=abs(targetVector[index] - vectorBeingCompared[index])
    return score

def calculateSimilarityScoreUsingL2(targetVector,vectorBeingCompared):
    score = 0.0
    for index in range(len(targetVector)):
        score+= math.sqrt(pow(targetVector[index],2)+ pow(vectorBeingCompared[index],2))
    return score

def calculateSimilarityScoreUsingCosine(targetVector,vectorBeingCompared):
    result = 1 - spatial.distance.cosine(targetVector, vectorBeingCompared)
    #result = dot(targetVector, vectorBeingCompared) / (norm(targetVector) * norm(vectorBeingCompared))
    return result

doc = xml.dom.minidom.parse("./data/devset_topics.xml")
titles = doc.getElementsByTagName('title')
indexes = doc.getElementsByTagName('number')

for i in range(len(indexes)):
    if (int)(sys.argv[1]) == (int)(indexes[i].firstChild.data):
        locationName = titles[i].firstChild.data
    otherLocationNames.append(titles[i].firstChild.data)

print "Target Location Name = "+str(locationName)
#print "Other Location Names = "+ str(otherLocationNames)
for file in os.listdir("./data/img/"):
    if locationName in file:
        targetFileNames.append("./data/img/"+file)
    fileNamesToCompare.append("./data/img/"+file)

print "Location names to compare = "+str(len(fileNamesToCompare))

#calculate the file clusters for each location
otherLocationFileCluster = []
for location in otherLocationNames:
    tempCluster=[]
    for fileName in fileNamesToCompare:
        if location in fileName:
            tempCluster.append(fileName)
    otherLocationFileCluster.append(tempCluster)

print "No of locations to compare = "+str(len(otherLocationFileCluster)) +" with "+str(len(otherLocationFileCluster[0]))+"files for each location"

targetReducedVector = returnSingleReducedVectorFOrAllModelsForLocation(targetFileNames)
allFileClusterVectors = []
for idx, fileCluster in enumerate(otherLocationFileCluster):
    allFileClusterVectors.append(returnSingleReducedVectorFOrAllModelsForLocation(fileCluster))

print "All files reduced to single vectors"

#calculate similarity scores between all vectors n allFileClusterVectors and targetReducedVector and finally return least 5 scores
scoresArr=[]
for idx, fileVector in enumerate(allFileClusterVectors):
    scoresArr.append(calculateSimilarityScoreUsingL1(targetReducedVector,fileVector))

print "Scores Calculated"
print scoresArr
print "Length = "+str(len(scoresArr))

#calculating top 5 scores and their locations
mostSimilarIndexes = sorted(range(len(scoresArr)), key=lambda i: scoresArr[i])[:(int)(sys.argv[2])]

#print "most similar indexes = "
#print mostSimilarIndexes


print "============ Most Similar 5 Locations for " + locationName +"=========="

for i in mostSimilarIndexes:
    print "Location = "+str(otherLocationNames[i])+" Score = "+str(scoresArr[i])


print "\n Total Time taken to Execute"
print str(datetime.datetime.now()-startTime)