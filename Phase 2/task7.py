
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

def getCSVDataAsListData(fileName):
    mainData = []
    with open(fileName) as csv_file:
        csvData = csv.reader(csv_file, delimiter=',')
        for row in csvData:
            mainData.append(row)
        return mainData

def returnUserTerms(row):
    col_count = 1
    tempTermArray = []
    while col_count < len(row):
        tempTermArray.append(row[col_count])
        col_count += 4
    tempTermArray.pop()
    return tempTermArray

def calculateStartingTermIndex(row):
    index = 0
    loacationNameWords = row[0].count('_')
    return loacationNameWords+2

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

userFileData = getCSVDataAsListData(userFileName)
imageFileData = getCSVDataAsListData(imageFileName)
locationFileData = getCSVDataAsListData(locationFileName)

print ("User FIle Data Length = "+str(len(userFileData)))
print ("image FIle Data Length = "+str(len(imageFileData)))
print ("location FIle Data Length = "+str(len(locationFileData)))

#originalDataTensor=[[[0]*len(userFileData)]*len(imageFileData)]*len(locationFileData)
originalDataTensor = np.full((len(locationFileData), len(imageFileData), len(userFileData)), 0)
for idx,location in enumerate(locationFileData):
    locationTerms = returnUserTermsForLocation(location)
    for idy,image in enumerate(imageFileData):
        imageTerms = returnUserTerms(image)
        for idz,user in enumerate(userFileData):
            userTerms = returnUserTerms(user)
            commonTermsArray = set(userTerms) & set(imageTerms) & set(locationTerms)
            originalDataTensor[idx][idy][idz] = len(commonTermsArray)

#pprint.pprint(originalDataTensor)
npArr = np.asarray(originalDataTensor)
#print(npArr)
#tensor = tl.tensor(originalDataTensor)
tensor = tl.tensor(npArr)
while (k!=0):
    factormatrix = parafac(tensor,rank=k)
    for i in range(0,3):
        print("============",i,"Factor Matrix","==============")
        print(factormatrix[i])
    print("If you wish to input another query, enter value of K else enter 0 to exit")
    k = int(input())
    
print ("\n Total Time taken to Execute")
print (str(datetime.datetime.now()-startTime))