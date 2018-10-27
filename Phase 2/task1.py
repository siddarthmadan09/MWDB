import sys
from pymongo import MongoClient
from datetime import datetime
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from numpy import linalg
import lda

import task1_2_base

startTime = datetime.now()

## input: python task1 user/image/location svd/pca/lda k

### Start
if len(sys.argv) == 4:
    dataType = str(sys.argv[1])
    decompositionMethod = str(sys.argv[2])
    k = int(sys.argv[3])

    dataArray, docs, terms = task1_2_base.computeDataArray(dataFamily=dataType)

    if decompositionMethod == "svd":
        termWeightPairs = task1_2_base.svd_reduction(dataArray, k)
        print (termWeightPairs)
    elif decompositionMethod == "pca":
        termWeightPairs = task1_2_base.pca_reduction(dataArray, k)
        print (termWeightPairs)
    elif decompositionMethod == "lda":
        topicWord, documentWord = task1_2_base.lda_reduction(dataArray, k)
        print (documentWord)
    else:
        print ("Invalid decomposition method")
        sys.exit(0)
else:
    print ("Invalid input")
    sys.exit(0)

print ("Total time taken: ", datetime.now() - startTime)