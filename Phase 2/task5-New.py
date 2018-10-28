import csv
import sys
import math
import os
import xml.dom.minidom
import datetime
import pandas as pd
import statistics
#from scipy.sparse import csr_matrix
#from sklearn.decomposition import TruncatedSVD
#from sklearn.random_projection import sparse_random_matrix
#import scipy.sparse
#from numpy.linalg import svd
from numpy.linalg import svd
import numpy as np
from scipy.sparse import lil_matrix
import lda
from sklearn.cluster import KMeans
from numpy import dot
from numpy.linalg import norm
#from sklearn.decomposition import PCA
from numpy.linalg import svd
from scipy import spatial
#from sklearn.decomposition import LatentDirichletAllocation
objectLSMatrix=[]
startTime = datetime.datetime.now()
fileName = ''
locationName = ""
otherLocationNames=[]
targetFileNames = []
fileNamesToCompare = []
imageIDArr=[]
def getCSVDataAsListData(fileName):
    mainData = []
    with open(fileName) as csv_file:
        csvData = csv.reader(csv_file, delimiter=',')
        for row in csvData:
            mainData.append(row)
        return mainData

def findSimiliarityDist(inputVector, currentVector):
    dist = sum([(a - b)** 2 for a, b in zip(inputVector, currentVector)])**(1/2)

    return dist

def calculateSimilarityScoreUsingL2(targetVector,vectorBeingCompared):
    score = 0.0
    for index in range(len(targetVector)):
        score+= pow(targetVector[index]-vectorBeingCompared[index],2)
    return math.sqrt(score)

def createAllModelMatrix(targetFileNames):
    finalVector = []
    #arrLength = len(getCSVDataAsListData(targetFileNames[0]))
    for x, file in enumerate(targetFileNames):
        #tempArr = []
        filedata = getCSVDataAsListData(file)
        print file
        print len(filedata)
        #if file == "./data/img/aztec_ruins LBP.csv":
        #    print "Problem"
        fileDataNP = np.asarray(filedata)
        deletedArr = np.delete(fileDataNP,[0],axis=1)
        if len(finalVector) == 0:
            finalVector = deletedArr
        else:
            finalVector = np.append(finalVector,deletedArr, axis=1)
        #allModalIntArray = np.asfarray(allModelVector, float)
    return  np.asfarray(finalVector, float)



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
for idx,location in enumerate(otherLocationNames):
    tempCluster=[]
    for idy,fileName in enumerate(fileNamesToCompare):
        if location in fileName:
            tempCluster.append(fileName)
    otherLocationFileCluster.append(tempCluster)

print "No of locations to compare = "+str(len(otherLocationFileCluster)) +" with "+str(len(otherLocationFileCluster[0]))+"files for each location"


allModalArray = createAllModelMatrix(targetFileNames)
#kmeans = KMeans(n_clusters=10, random_state=0).fit(allModalArray)
allModalIntArray = allModalArray #kmeans.cluster_centers_

if sys.argv[3].upper() == "SVD":
    U, singularValues, V = svd(allModalIntArray,full_matrices=False)
    reducedArr = V[:(int)(sys.argv[2]), :]
    transVMatrix = np.transpose(reducedArr)
    lsMatrix = np.dot(allModalIntArray, transVMatrix)
elif sys.argv[3].upper() == "PCA":
    covMatrix = np.cov(allModalIntArray)
    U, singularValues, V = svd(covMatrix, full_matrices=False)
    reducedArr = V[:(int)(sys.argv[2]), :]
    transVMatrix = np.transpose(reducedArr)
    lsMatrix = np.dot(covMatrix, transVMatrix)
elif sys.argv[3].upper() == "LDA":
    print "Incorrect Model Entered.. Try Again"
    sys.exit()
#reducedArr = V[:(int)(sys.argv[2]), :]
print "3rd matrix - Reduced"
            # print reducedArr
print "Rows = " + str(len(reducedArr)) + " Out of " + str(len(V))
print "Columns = " + str(len(reducedArr[0])) + " Out of " + str(len(V[0]))

#adding IMageIDs to matrix

allData = getCSVDataAsListData(targetFileNames[0])
npArrayTemp = np.asarray(allData)
ImageIdArr = npArrayTemp[:,:1]
#print "imageIds"
#print ImageIdArr
wholeMatrix = np.append(ImageIdArr,lsMatrix,axis=1)
#imageIds = mat(:,2);
print "LS matrix calulated"
dataframe = pd.DataFrame(data=wholeMatrix.astype(float))
dataframe.to_csv('outfile_'+sys.argv[3].upper()+"_"+str(datetime.datetime.now())+'.csv', sep=' ', header=False, float_format='%.2f', index=False)
print "file save done"

#
#PART - 2
#

#input location in clusters
kmeans = KMeans(n_clusters=5, random_state=0).fit(allModalArray)
allModalIntArray = kmeans.cluster_centers_

if sys.argv[3].upper() == "SVD":
    U, singularValues, V = svd(allModalIntArray,full_matrices=False)
    reducedArr = V[:(int)(sys.argv[2]), :]
    transVMatrix = np.transpose(reducedArr)
    targLSMatrix = np.dot(allModalIntArray, transVMatrix)
elif sys.argv[3].upper() == "PCA":
    covMatrix = np.cov(allModalIntArray)
    U, singularValues, V = svd(covMatrix, full_matrices=False)
    reducedArr = V[:(int)(sys.argv[2]), :]
    transVMatrix = np.transpose(reducedArr)
    targLSMatrix = np.dot(covMatrix, transVMatrix)
elif sys.argv[3].upper() == "LDA":
    print "Incorrect Model Entered.. Try Again"
    sys.exit()

locScores=[]
for idx, fileCluster in enumerate(otherLocationFileCluster):
    #print fileCluster[0]
    allModalArray = createAllModelMatrix(fileCluster)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(allModalArray)
    allModalIntArray = kmeans.cluster_centers_
    if sys.argv[3].upper() == "SVD":
        U, singularValues, V = svd(allModalIntArray, full_matrices=False)
        reducedArr = V[:(int)(sys.argv[2]), :]
        transVMatrix = np.transpose(reducedArr)
        lsMatrix = np.dot(allModalIntArray, transVMatrix)
    elif sys.argv[3].upper() == "PCA":
        covMatrix = np.cov(allModalIntArray)
        U, singularValues, V = svd(covMatrix, full_matrices=False)
        reducedArr = V[:(int)(sys.argv[2]), :]
        transVMatrix = np.transpose(reducedArr)
        lsMatrix = np.dot(covMatrix, transVMatrix)
    elif sys.argv[3].upper() == "LDA":
        print "Incorrect Model Entered.. Try Again"
        sys.exit()
    #comparing LS matrix for this location to target location
    for i, v1 in enumerate(targLSMatrix):
        interClusDist = []
        min_dist = float("inf")
        for j, v2 in enumerate(lsMatrix):
            min_dist = min(min_dist, calculateSimilarityScoreUsingL2(v1, v2))
        interClusDist.append(min_dist)
    locScores.append(statistics.mean(interClusDist))

#print "Location Scores"
#print locScores

mostSimilarIndexes = sorted(range(len(locScores)), key=lambda i: locScores[i])[:(int)(sys.argv[2])]

print "============ Most Similar 5 Locations for " + locationName +"=========="

for i in mostSimilarIndexes:
    print "Location = "+str(otherLocationNames[i])+" Score = "+str(locScores[i])
# targetReducedVector = returnSingleReducedVectorFOrAllModelsForLocation(targetFileNames)
# allFileClusterVectors = []
# for idx, fileCluster in enumerate(otherLocationFileCluster):
#     allFileClusterVectors.append(returnSingleReducedVectorFOrAllModelsForLocation(fileCluster))
#
# print "All files reduced to single vectors"
#
# #calculate similarity scores between all vectors n allFileClusterVectors and targetReducedVector and finally return least 5 scores
# scoresArr=[]
# for idx, fileVector in enumerate(allFileClusterVectors):
#     scoresArr.append(calculateSimilarityScoreUsingL2(targetReducedVector,fileVector))
#
# print "Scores Calculated"
# print scoresArr
# print "Length = "+str(len(scoresArr))

#calculating top 5 scores and their locations
#mostSimilarIndexes = sorted(range(len(scoresArr)), key=lambda i: scoresArr[i])[:(int)(sys.argv[2])]

#print "most similar indexes = "
#print mostSimilarIndexes


# print "============ Most Similar 5 Locations for " + locationName +"=========="
#
# for i in mostSimilarIndexes:
#     print "Location = "+str(otherLocationNames[i])+" Score = "+str(scoresArr[i])

#
# print "OBJECT MATRIX"
# print objectLSMatrix

print "\n Total Time taken to Execute"
print str(datetime.datetime.now()-startTime)