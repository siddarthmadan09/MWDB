import csv
import sys
import math
import datetime
import numpy as np
import webbrowser

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags

def getIndexOfStartVectors(allImageIDs, startVectors):
    startVectorIndices = []
    for seed in startVectors:
        if seed in allImageIDs:
            startVectorIndices.append(allImageIDs.index(seed))
    return startVectorIndices    

def personalizedPageRank(imgImgArr, startVectors, c=0.15, allowedDiff=1e-9, maxIters = 100):
    
    # convert to (sparse) adjacency matrix
    sparseImgImgArray = csr_matrix(imgImgArr)

    # row normalize adjacency matrix
    m, n = sparseImgImgArray.shape
    d = sparseImgImgArray.sum(axis=1)

    # handling 0 entries
    d = np.maximum(d, np.ones((n, 1)))
    invd = 1.0 / d
    invd = np.reshape(invd, (1,-1))
    invD = spdiags(invd, 0, m, n)

    # row normalized adj. mat. 
    rowNormSparseImgImgArray = invD * sparseImgImgArray 
    rowNormSparseImgImgArrayTranspose = rowNormSparseImgImgArray.T
    
    # init seed vectors
    seedVectors = np.zeros((n, 1))
    seedVectors[startVectors] = 1.0/len(startVectors)
    
    # init PPR
    cuurent_nodes = seedVectors
    old_nodes = seedVectors
    diff = np.zeros((maxIters, 1))

    # iterate
    for i in range(maxIters):
        cuurent_nodes = (1-c)*(rowNormSparseImgImgArrayTranspose.dot(old_nodes)) + c*seedVectors

        diff[i] = norm(cuurent_nodes - old_nodes, 1)       
        if diff[i] <= allowedDiff:
           break

        old_nodes = cuurent_nodes
    
    # find and return top k
    return cuurent_nodes


def classify(similarityMatrix, trainingSet, allImageIDs):
    pprScoresForEachLabel = []
    # prep training data and labels
    trainingData = {}
    for pair in trainingSet.items():
        if pair[1] not in trainingData.keys():
            trainingData[pair[1]] = []

        trainingData[pair[1]].append(pair[0])

    labels = list(trainingData.keys())

    for label, seeds in trainingData.items():
        startVectorIndices = getIndexOfStartVectors(allImageIDs, seeds)
        pprScores = personalizedPageRank(similarityMatrix, startVectorIndices, maxIters = 100)

        if len(pprScoresForEachLabel) == 0:
            pprScoresForEachLabel = pprScores
        else:
            pprScoresForEachLabel = np.hstack((pprScoresForEachLabel, pprScores))

    labelIndices = (np.argmax(pprScoresForEachLabel, axis=1)).tolist()
    
    imgLabelPairs = []
    # for i in range (len(allImageIDs)):
    #     imgLabelPairs.append({labels[int(labelIndices[i])]:allImageIDs[i]})
    for i in range (len(allImageIDs)):
        imgLabelPairs.append({
            'imageId': allImageIDs[i],
            'label': labels[int(labelIndices[i])],
            'score': pprScoresForEachLabel[i][labelIndices[i]]
            })

    imgLabelPairs = sorted(imgLabelPairs, key=lambda k: k['score'], reverse=True)

    imgLabels = {}
    # for each in imgLabelPairs:
    #     for pair in each.items():
    #         if pair[0] not in imgLabels.keys():
    #             imgLabels[pair[0]] = []

    #         imgLabels[pair[0]].append(pair[1])
    for each in imgLabelPairs:
        if each['label'] not in imgLabels.keys():
            imgLabels[each['label']] = []
        imgLabels[each['label']].append(each['imageId'])

    return imgLabels


def showImagesInWebPageForPPR(imgPaths):
    print("\n Creating Web Page")
    f = open('task4PPRoutput.html', 'w')
    message = """<html><head></head><body>"""
    f.write(message)
    # add html code here
    content = """<table><tbody>"""
    for idx, imgData in enumerate(imgPaths):
        if idx == 0:
            content = content + """<tr><td><h2>Seed Vectors</h2></td></tr><tr>"""
        if idx == 3:
            content = content + """</tr><tr></tr><tr><td><h2>PPR results</h2></td></tr>"""
        for imgId, imgFilePath in imgData.items():
            content = content + """<td><figure><img src=\"""" + imgFilePath + """\" height="400" width="100%"><figcaption>""" + imgId + """</figcaption></figure></td>"""
        if idx - 3 > 0 and (idx - 3) % 4 == 0:
            content = content + """</tr><tr>"""
    content = content + """</tr>"""

    content = content + """</tbody></table>"""
    f.write(content)
    f.write("""</body></html>""")
    f.close()
    filename = 'file:///home/leroy/Documents/CSE 515/Phase 3/task4PPRoutput.html'
    webbrowser.open_new_tab(filename)