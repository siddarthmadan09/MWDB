import csv
import random
import sys
import time
import math
import os
import xml.dom.minidom
from numpy.linalg import svd
import numpy as np
from sklearn.preprocessing import Normalizer, MinMaxScaler
from task5 import LSHImpl, EuclideanFamily
import webbrowser
import datetime

path = "./data/img/"

def copyFiles(filename) :
    """returns the file names of the images """
    path = "./data/images/"
    for path, subdirs, files in os.walk(path):
        for name in files:
            if (filename in name):
                #print("match")
                #shutil.copy(os.path.join(path, name), dirname)
                return os.path.join(path, name)

def showImagesInWebPage(clusterDict,imageIds,k, l,d,w):
    """display the similar images in a  web page"""
    f = open('task5output.html', 'w')
    message = """<html><head></head><body>"""
    f.write(message)
    # add html code here
    content = """<table><tbody>"""

    content = content + """<tr><td><H1> L:""" + l +""" K:  """  + k +   """ D:  """ + d +  """ W:  """ + w + """</H1></td></tr><tr>"""
    for idx, image in enumerate(clusterDict):
        if not image == None and ".jpg" in image:
            content = content + """<th>"""+imageIds[idx]+"""</th><td><img src=\"""" + image + """\" height="100" width="100"></td>"""
    content = content + """</tr>"""

    content = content + """</tbody></table>"""
    f.write(content)
    f.write("""</body></html>""")
    f.close()

def computeSvd(X,k):
    """perform dimensionality reduction on the input vectors"""
    np_X = np.asarray(X, dtype = float) # CONVERTING locationImageData to np_locationImageData
    U, s, Vt = svd(X,full_matrices=False)
    Vt = Vt[:k,:]
    return np.dot(np_X, np.transpose(Vt))

def getImageIndex(allImageIDs,q):
        for i, imageID in enumerate(allImageIDs):
            if imageID in q:
                index  = i
                break
        return index

def getCSVDataAsListData(fileName):
    mainData = []
    with open(fileName) as csv_file:
        csvData = csv.reader(csv_file, delimiter=',')
        for row in csvData:
            mainData.append(row)
        return mainData

def createAllModelMatrix(targetFileNames):
    """concatenate all the visual models for a location and perform normalization"""
    finalVector = []
    for x, file in enumerate(targetFileNames):
        filedata = getCSVDataAsListData(file)
        fileDataNP = np.asarray(filedata)
        deletedArr = np.delete(fileDataNP,[0],axis=1)
        if len(finalVector) == 0:
            finalVector = deletedArr
        else:
            finalVector = np.append(finalVector,deletedArr, axis=1)
            finalVector = finalVector.astype(np.float64)
            # scaler = MinMaxScaler()
            # scaler.fit(finalVector)
            # finalVector = scaler.transform(finalVector) * 1000
    finalVector = finalVector.astype(np.float64)
    transformer = Normalizer().fit(finalVector) # fit does nothing.
    finalVector = transformer.transform(finalVector)
    # scaler = MinMaxScaler()
    # scaler.fit(finalVector)
    # finalVector = scaler.transform(finalVector) * 1000
    return  (np.asfarray(finalVector, float)), fileDataNP[:,0]


def main():
    startTime = datetime.datetime.now()
    allImageIDs = []
    doc = xml.dom.minidom.parse("./data/devset_topics.xml")
    titles = doc.getElementsByTagName('title')
    indexes = doc.getElementsByTagName('number')
    locationNames = []
    for i in range(len(indexes)):
            locationNames.append(titles[i].firstChild.data)
    targetFileNames = []


    files = []
    for locationName in locationNames:
        files = []
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path,i)) and locationName in i:
                files.append(path + i)
        targetFileNames.append(files)
    count = 0
    finalImageFeatureMatrix = []

    for x,files in enumerate(targetFileNames):
        allModalArray,imageIdArr = createAllModelMatrix(files)
        if len(finalImageFeatureMatrix) == 0:
            finalImageFeatureMatrix = allModalArray
            allImageIDs.extend(imageIdArr)
        else:
            finalImageFeatureMatrix = np.append(finalImageFeatureMatrix, allModalArray, axis=0)
            allImageIDs.extend(imageIdArr)
    scaler = MinMaxScaler()
    scaler.fit(finalImageFeatureMatrix)
    #finalImageFeatureMatrix = scaler.transform(finalImageFeatureMatrix) * 100000000000
    finalImageFeatureMatrix = finalImageFeatureMatrix.astype(np.float64)
    transformer = Normalizer().fit(finalImageFeatureMatrix) # fit does nothing.
    finalVector = transformer.transform(finalImageFeatureMatrix)
    sv = computeSvd(finalImageFeatureMatrix,450)
    #sv = finalImageFeatureMatrix
    l = int(input("Enter l"))
    k = int(input("Enter k "))

    num_neighbours = 2
    radius = 0.1

    # for i,point in enumerate(sv):
    #     for i in range(num_neighbours):
    #         newvec.append([x+random.uniform(-radius,radius) for x in point])
    #         imageid.append(allImageIDs[i])
    # new = np.asarray(newvec)
    # imageids = np.asarray(imageid)
    # np.asarray(allImageIDs)
    # imageids = allImageIDs + imageid
    # print((new.shape))
    # sv = np.append(sv,new,axis=0)

    (r,d ) = sv.shape
    print(d)

    hashFamily = EuclideanFamily(l,k,d)

    lsh = LSHImpl(d,k,l,hashFamily)
    buckets = lsh.reindex(sv,allImageIDs)

    #queryPoint = '10195754275'
    c = "c"
    while c != "q":
        queryPoint = input("Enter query id:")
        index = 0

        index = getImageIndex(allImageIDs,queryPoint)
        print(index)
        points = sv[index]
        q = int(input("Enter t for t similar images:"))

        candidates = []
        queriedHashes, listofTuples = lsh.query(points,sv,allImageIDs,buckets,q)
        matches = ([x[0] for x in listofTuples])
        print(listofTuples)
        candidates.append(matches)
        flat_list = [item for sublist in candidates for item in sublist]
        iterations = 0
        while len(flat_list) < q and iterations < 4:
            iterations += 1
            remaining = q - len(flat_list)
            listofTuples = set(lsh.requery(queryPoint,sv,allImageIDs,buckets, queriedHashes, remaining))
            print(listofTuples)
            matches = ([x[0] for x in listofTuples])
            candidates.append(matches)
        flat_list = [item for sublist in candidates for item in sublist]
        flat_list = flat_list[:q]
        clusterDict = []
        print('main flat  ', flat_list)

        for val in flat_list:
            val =  ''.join(val)
            clusterDict.append(copyFiles( val + ".jpg"))
        print('clsuterdcir  ' , clusterDict)
        showImagesInWebPage(clusterDict,flat_list,str(k),str(l),str(d), str(0.5))
        print (str(datetime.datetime.now()-startTime))
        c = input("enter c to continue or q to quit")


if __name__ == "__main__":
    main()




