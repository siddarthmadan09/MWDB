
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
import pprint

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


userFileName = './testdata/devset_textTermsPerUser.csv'
locationFileName = './testdata/devset_textTermsPerPOI.wFolderNames.csv'
imageFileName = './testdata/devset_textTermsPerImage.csv'

userFileData = getCSVDataAsListData(userFileName)
imageFileData = getCSVDataAsListData(imageFileName)
locationFileData = getCSVDataAsListData(locationFileName)

print "User FIle Data Length = "+str(len(userFileData))
print "image FIle Data Length = "+str(len(imageFileData))
print "location FIle Data Length = "+str(len(locationFileData))

originalDataTensor=[[[0]*len(userFileData)]*len(imageFileData)]*len(locationFileData)
val=0
counter=0
for idx,location in enumerate(locationFileData):
    locationTerms = returnUserTermsForLocation(location)
    for idy,image in enumerate(imageFileData):
        imageTerms = returnUserTerms(image)
        for idz,user in enumerate(userFileData):
            userTerms = returnUserTerms(user)
            commonTermsArray = set(userTerms) & set(imageTerms) & set(locationTerms)
            counter=counter+1
            print (counter)
            originalDataTensor[idx][idy][idz] = len(commonTermsArray)
            originalDataTensor[idx][idy][idz] = val
            val = (val+1)%2

pprint.pprint(originalDataTensor)
#npArr = np.array(originalDataTensor)

tensor = tl.tensor(originalDataTensor)
#tensor2 = tl.tensor(npArr)

print "Tensor"
print tensor
print "single value"
print tensor[0][0][0]
#tensor[0][0][0] = 1.
#tensor[0][0][1] = 1.
#tensor[1][1][0] = 1.
#print "Tensor Updated"
#print tensor2

print parafac(originalDataTensor,rank=5)
print "\n Total Time taken to Execute"
print str(datetime.datetime.now()-startTime)