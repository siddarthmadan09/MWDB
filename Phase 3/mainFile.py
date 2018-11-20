import csv
import sys
import math
import datetime
import numpy as np
import random
import gc
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy
import operator
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity as cosineSimilarity
from sklearn.metrics.pairwise import euclidean_distances as euclideanDistance
from sklearn.metrics.pairwise import manhattan_distances as manhattanDistance
def returnUserTerms(row):
    col_count = 1
    tempTermArray = []
    while col_count < len(row):
        tempTermArray.append(row[col_count])
        col_count += 4
    tempTermArray.pop()
    return tempTermArray

def getCSVDataAsListData(fileName):
        mainData = []
        with open(fileName) as csv_file:
            csvData = csv.reader(csv_file, delimiter=',')
            for row in csvData:
                mainData.append(row)
            return mainData

def returnUserTfValues(row):
    col_count = 2
    tempTermArray = []
    while col_count < len(row):
        tempTermArray.append((int)(row[col_count]))
        col_count += 4
    return tempTermArray

def returnUserDfValues(row):
        col_count = 3
        tempTermArray = []
        while col_count < len(row):
            tempTermArray.append(row[col_count])
            col_count += 4
        return tempTermArray

def returnUserTfIdfValues(row):
        col_count = 4
        tempTermArray = []
        while col_count < len(row):
            tempTermArray.append(row[col_count])
            col_count += 4
        return tempTermArray

def getAllImagesTermDict(fileName):
    allImageTermDict={}
    allImageData = getCSVDataAsListData(fileName)
    for row in allImageData:
        terms = returnUserTerms(row)
        allImageTermDict[row[0]] = terms
    return allImageTermDict

def getAllImagesTFDict(fileName):
    allImageTFDict={}
    allImageData = getCSVDataAsListData(fileName)
    for row in allImageData:
        terms = returnUserTfValues(row)
        allImageTFDict[row[0]] = terms
    return allImageTFDict

def createDict(mainTermArr,mainTFValues):
    dict={}
    for idx,term in enumerate(mainTermArr):
        dict[term] = mainTFValues[idx]
    return dict

def findSimilarityScoreBetweenImages(mainDict,otherDict):
    score = 0
    for key in mainDict:
        if key in otherDict:
            score = score + abs(mainDict[key]-otherDict[key])
        else:
            score = score + abs(mainDict[key])
    #all(map(mainDict.pop, otherDict))
    for key in otherDict:
        if key not in mainDict:
            score = score +abs(otherDict[key])
    return score

def findSimilarityScoreBetweenImagesUsingL2(mainDict,otherDict):
    score = 0
    for key in mainDict:
        if key in otherDict:
            score = score + pow(mainDict[key]-otherDict[key],2)
        else:
            score = score + pow(mainDict[key],2)
    #all(map(mainDict.pop, otherDict))
    for key in otherDict:
        if key not in mainDict:
            score = score +pow(otherDict[key],2)
    return math.sqrt(score)


#get 5 most similar images to given image.
def getKMostSimilarImagesAndScores(scoreArr,k):
    a = np.asarray(scoreArr)
    #imageIDArray = np.asarray(allImageIDs)
    imageIDArray = np.asarray(imageIDList)
    idx = np.argpartition(a, k)
    #keys = list(imageTermDict.keys())
    imageIDs = imageIDArray[idx[:k]]
    scores = scoreArr[idx[:k]]
    return imageIDs,scoreArr

def getKMostSimilarImagesAndScoresAsDict(scoreArr,k):
    a = np.asarray(scoreArr)
    #imageIDArray = np.asarray(allImageIDs)
    imageIDArray = np.asarray(imageIDList)
    idx = np.argpartition(a, k)
    #keys = list(imageTermDict.keys())
    imageIDs = imageIDArray[idx[:k]]
    scores = scoreArr[idx[:k]]
    tempDict={}
    for idx,imageid in enumerate(imageIDs):
        tempDict[imageid] = scores[idx]
    return tempDict

def getAllSimilarImagesAndScoresAsDict(scoreArr):
    tempDict={}
    for idx,imageid in enumerate(imageIDList):
        tempDict[imageid] = scoreArr[idx]
    return tempDict

def printAndSaveGraphProperly(grph,name):
    with open(name, 'w') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in grph.items()]

def createClusterDict(listofclusters):
    clusterDict = {}
    for centroid in listofclusters:
        tempDict = {}
        clusterDict[centroid] = tempDict
    for key in outputDict:
        minDistance = 999999999
        closestClusterCentroid = ""
        for centroid in listofclusters:
            if minDistance > outputDict[key][centroid]:
                minDistance = outputDict[key][centroid]
                closestClusterCentroid = centroid
        clusterDict[closestClusterCentroid][key] = outputDict[key][closestClusterCentroid]
    return clusterDict

def getAllUniqueTerms(fileName):
    allImageData = getCSVDataAsListData(fileName)
    allterms=[]
    for row in allImageData:
        imageIDList.append(row[0])
        allterms.extend(returnUserTerms(row))
    return set(allterms)

startTime = datetime.datetime.now()
fileName = './data/devset_textTermsPerImage.csv'
allImageData = getCSVDataAsListData(fileName)
imageTermDict = getAllImagesTermDict(fileName)
imageTFDict = getAllImagesTFDict(fileName)
allImageIDs = list(imageTermDict.keys())
count=0;
imageIDList=[]
allUniqueTerms = getAllUniqueTerms(fileName)
print("\nunique terms calculated")
print ("Total Time taken to Execute")
print (str(datetime.datetime.now()-startTime))
startTime = datetime.datetime.now()

imageTermMatrix = numpy.zeros((len(imageIDList), len(allUniqueTerms)))
for idx,row in enumerate(allImageData):
    userTerms = returnUserTerms(row)
    userTFIDFValues = returnUserTfIdfValues(row)
    for idy,term in enumerate(userTerms):
        index = term in allUniqueTerms
        imageTermMatrix[idx][index] = userTFIDFValues[idy]

print ("\nimage term matrix created")
print ("Total Time taken to Execute")
print (str(datetime.datetime.now()-startTime))
startTime = datetime.datetime.now()

# similarityMatrix = euclideanDistance(imageTermMatrix)
similarityMatrix = cosineSimilarity(imageTermMatrix)
# similarityMatrix = manhattanDistance(imageTermMatrix)
#gc.collect()
print("\ncosine similarty done")
print ("Total Time taken to Execute")
print (str(datetime.datetime.now()-startTime))
startTime = datetime.datetime.now()
# totalDict={}
# for idx,row in enumerate(similarityMatrix):
#     totalDict[imageIDList[idx]] = getAllSimilarImagesAndScoresAsDict(row)
# # pickle.dump( totalDict, open( "allImageSimilarityNXGraph3-pickle.p", "wb" ) )
# print("\nEntire image image graph done")
# print ("Total Time taken to Execute")
# print (str(datetime.datetime.now()-startTime))
# startTime = datetime.datetime.now()
# gc.collect()
# printAndSaveGraphProperly(totalDict,"AllImageGraph.csv")

# print("\nimage image graph saved as csv")
# print ("Total Time taken to Execute")
# print (str(datetime.datetime.now()-startTime))
# startTime = datetime.datetime.now()
outputDict={}
G = nx.DiGraph()
taskNumber = (int)(input("Enter task number = "))
while taskNumber>0:

    if taskNumber == 1:
        k = (int)(input("Enter value for K = "))
        #calculating most similar k values for each image
        # G = nx.Graph()

        for idx,row in enumerate(similarityMatrix):
            outputDict[imageIDList[idx]] = getKMostSimilarImagesAndScoresAsDict(row,k)
            imageIDs,scoreIDs = getKMostSimilarImagesAndScores(row,k)
            for idx,imageID in enumerate(imageIDs):
                G.add_edge(imageIDList[idx], imageID, capacity=scoreIDs[idx])
        printAndSaveGraphProperly(outputDict,"task1-output.csv")

        print ("\nTask 1 complete")
        print ("Total Time taken to Execute")
        print (str(datetime.datetime.now()-startTime))

    elif taskNumber == 2:

        print("Task 2 code here")
        c = (int)(input("Enter number of Clusters c = "))

        # finding initial c centroid points
        listofclusters = []
        # c1 = random.choice(list(outputDict.keys()))
        # listofclusters.append(c1)
        listofclusters = random.sample(list(outputDict.keys()),c)
        # A = Counter(outputDict[c1])
        # for i in range(1, c):
        #     for pt in listofclusters:
        #         del A[pt]
        #     cx = max(A.items(), key=operator.itemgetter(1))[0]
        #     listofclusters.append(cx)
        #     B = Counter(outputDict[cx])
        #     A = A + B
        # print(listofclusters)
        clusterDict={}
        # for centroid in listofclusters:
        #     tempDict = {}
        #     clusterDict[centroid] = tempDict
        # #creating clusters with the above points as centroids
        # for key in totalDict:
        #     minDistance = 999999999
        #     closestClusterCentroid = ""
        #     for centroid in listofclusters:
        #         if minDistance > totalDict[key][centroid]:
        #             minDistance = totalDict[key][centroid]
        #             closestClusterCentroid = centroid
        #     clusterDict[closestClusterCentroid][key]=totalDict[key][closestClusterCentroid]
        clusterDict = createClusterDict(listofclusters)
        #now find a random point inside the previously created  clusters and run the above algo again

    elif taskNumber == 3:
        print("Task 3 code here")

    elif taskNumber == 4:
        print("Task 4 code here")

    elif taskNumber == 5:
        print("Task 5 code here")

    elif taskNumber == 6:
        print("Task 5 code here")

    taskNumber = (int)(input("Enter task number = "))