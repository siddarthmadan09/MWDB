# -*- coding: utf-8 -*-

import csv
import sys
import numpy as np
import operator
from collections import defaultdict
from xml.dom.minidom import parse
import xml.dom.minidom

# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse(r"C:/Users/vagarw14/mwdb3/devset_topics.xml")
collection = DOMTree.documentElement
locationImageData = []     # A list to store all images data vectors for one location
location_name_dict = {}
# CODE TO GET FILENAME OF ALL  LOCATION FILES
def getLocationNames():     
    topics = collection.getElementsByTagName("topic")
    for topic in topics:
        location_name_dict[int(topic.getElementsByTagName('number')[0].childNodes[0].data)] = topic.getElementsByTagName('title')[0].childNodes[0].data
    return location_name_dict


# print(location_name_dict)

# RIGHT NOW LOCATION ID IS CONVERTED TO INT inside dictionary


def readInput():
    loc_number= int(input('Enter location_number\n'))
    model = input('Enter the model of preference\n')
    kbest= int(input('K \n'))

def readInputImg():
    img_id = int(input('Enter image id:\n'))
    model = input('Enter the model of preference\n')
    kbest = int(input('K \n'))
    

def getImagedataByLoc(locationId, model):         
    arr=[]
    imageIds=[]                    # A list of all image IDS for each location
    filenameX = location_name_dict[locationId] + ' ' + str(model)
    # print(filenameX)
    # CODE TO READ CSV LOCATION_MODEL FILE FOR LOCATION GIVEN
    with open("C:/Users/vagarw14/mwdb3/descvis/img/"+filenameX+".csv","rt",newline='', encoding="utf8") as fp:
        
        line = fp.readline()
        while line:
            #arr = list(line.split(","))
            arr = line.split(",")
            arr[-1] = arr[-1][:-1]
            arr= [round(float(x),3) for x in arr]
            imageIds.append(int(arr[0]))
            locationImageData.append(arr)

            line = fp.readline()
#            locationAndImageId.append(locationX)
    
def computeSvd(getImagedataByLoc):
    U, s, Vt = svd(UserTermArr,full_matrices=False)
    print "SVD indivisual"
    print(U)
    print(singularValues)
    print(V)
    
    transArr = np.transpose(UserTermArr)
    
    termWeightMatrix = np.dot(transArr,U)
    
    print "Term Weight Array"
    print termWeightMatrix
    
    print "Term Weight Array length"
    print len(termWeightMatrix)
    
    print "Term Weight Array length for 1st row"
    print len(termWeightMatrix[0])

def main():
#    readInput()
#    readInputImg()
    i=0
    getLocationNames()
    for locationId in location_name_dict.keys() :
        getImagedataByLoc(locationId, "CM")
    
#    print(np.array(locationImageData).shape)
    
    
main()
    
    