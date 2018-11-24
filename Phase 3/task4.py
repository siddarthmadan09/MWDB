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

def personalizedPageRank(imgImgArr, startVectors, k, c=0.15, epsilon=1e-6, max_iters = 100):
    
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
    
    q = np.zeros((n, 1))
    q[startVectors] = 1.0/len(startVectors)
    
    x = q
    old_x = q
    residuals = np.zeros((max_iters, 1))

    for i in range(max_iters):
        x = (1-c)*(rowNormSparseImgImgArrayTranspose.dot(old_x)) + c*q

        residuals[i] = norm(x - old_x, 1)
       
        if residuals[i] <= epsilon:
           break

        old_x = x
    
    topkIndices = x.argsort(axis=0)[-k:][::-1]
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