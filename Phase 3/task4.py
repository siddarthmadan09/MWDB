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
    startVectorIndices.append(allImageIDs.index(startVectors[0]))
    startVectorIndices.append(allImageIDs.index(startVectors[1]))
    startVectorIndices.append(allImageIDs.index(startVectors[2]))
    return startVectorIndices    

def personalizedPageRank(imgImgArr, startVectors, k, c=0.15, allowedDiff=1e-6, maxIters = 100):
    
    # convert to (sparse) adjacency matrix
    sparseImgImgArray = csr_matrix(imgImgArr)

    # row normalize adjacency matrix
    m, n = sparseImgImgArray.shape
    rowNormSparseImgImgArrayTranspose = sparseImgImgArray.T
    
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
    topkIndices = cuurent_nodes.argsort(axis=0)[-k:][::-1]
    return topkIndices

def showImagesInWebPageForPPR(imgPaths):
    print("\n Creating Web Page")
    f = open('pprout.html', 'w')
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
    filename = 'file:///home/leroy/Documents/CSE 515/Phase 3/' + 'pprout.html'
    webbrowser.open_new_tab(filename)