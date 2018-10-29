
# find all commmon terms accross users, locations and images.
#create a matrix for user , location and images
# get the indivisual values as L2 distance of all three values above

import csv
import sys
import math
import os
import xml.dom.minidom
import datetime
import numpy as np
import scipy.sparse
from sklearn.cluster import KMeans
#from scipy import spatials
#import pprint

import tensorly as tl
from tensorly.decomposition import parafac

startTime = datetime.datetime.now()

#function to convert all entries in a CSV File into a list
def getCSVDataAsListData(fileName):
    mainData = []
    with open(fileName) as csv_file:
        csvData = csv.reader(csv_file, delimiter=',')
        for row in csvData:
            mainData.append(row)
        return mainData

#function which takes input objects (either users or images), collects the Terms in each row and returns it
def returnUserTerms(row):
    col_count = 1
    tempTermArray = []
    while col_count < len(row):
        tempTermArray.append(row[col_count])
        col_count += 4
    tempTermArray.pop()
    return tempTermArray

#function that calculates the sindex of the starting term for any given row(list of data)
def calculateStartingTermIndex(row):
    index = 0
    loacationNameWords = row[0].count('_')
    return loacationNameWords+2

#unction which takes input objects (locations), collects the Terms in each row and returns it
def returnUserTermsForLocation(row):
    col_count = calculateStartingTermIndex(row)
    tempTermArray = []
    while col_count < len(row):
        tempTermArray.append(row[col_count])
        col_count += 4
    tempTermArray.pop()
    return tempTermArray

k = int(sys.argv[1])
userFileName = '/home/vivek/Documents/MWDB_Phase2/Phase 2/testdata/devset_textTermsPerUser.csv'
locationFileName = '/home/vivek/Documents/MWDB_Phase2/Phase 2/testdata/devset_textTermsPerPOI.wFolderNames.csv'
imageFileName = '/home/vivek/Documents/MWDB_Phase2/Phase 2/testdata/devset_textTermsPerImage.csv'

userFileData = getCSVDataAsListData(userFileName)#converting user csv file data into a list
imageFileData = getCSVDataAsListData(imageFileName)#converting image csv file data into a list
locationFileData = getCSVDataAsListData(locationFileName)#converting location csv file data into a list

print ("User FIle Data Length = "+str(len(userFileData)))
print ("image FIle Data Length = "+str(len(imageFileData)))
print ("location FIle Data Length = "+str(len(locationFileData)))

#originalDataTensor=[[[0]*len(userFileData)]*len(imageFileData)]*len(locationFileData)
originalDataTensor = np.full((len(locationFileData), len(imageFileData), len(userFileData)), 0)#initializing the 3x3x3 matrix with all 0's intially
for idx,location in enumerate(locationFileData):
    locationTerms = returnUserTermsForLocation(location)
    for idy,image in enumerate(imageFileData):
        imageTerms = returnUserTerms(image)
        for idz,user in enumerate(userFileData):
            userTerms = returnUserTerms(user)
            commonTermsArray = set(userTerms) & set(imageTerms) & set(locationTerms)#finding intersection between userterms, imageterms and locationterms
            originalDataTensor[idx][idy][idz] = len(commonTermsArray)#appending the count of similar terms into the 3D matrix

tensor = tl.tensor(originalDataTensor)
while (k!=0):
    factormatrix = parafac(tensor,rank=k)#CP Decomposition. Here The array : factormatrix has 2 sub-arrays. Each containing one factor matrix for users, images and terms
    for i in range(0,3):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(factormatrix[i])#using k-means clustering to reduce overlapping amongst each factor matrix
        matrixToUse = kmeans.cluster_centers_
        print("============",i,"Clustered Factor Matrix","==============")
        print(matrixToUse)
    
    print("If you wish to input another query, enter value of K else enter 0 to exit")
    k = int(input())
    
print ("\n Total Time taken to Execute")
print (str(datetime.datetime.now()-startTime))