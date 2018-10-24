from pymongo import MongoClient
from datetime import datetime
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from numpy import linalg
import gensim

startTime = datetime.now()

def svd_reduction(array):
    global npUserTermArray
    print ("SVD::::::::")
    U, S, V = linalg.svd(npUserTermArray, full_matrices=False)

    # print (U)
    # print ('---------------------------------------------')
    # print (S)
    # print ('---------------------------------------------')
    # print (V)
    # print ("::::::::::::")
    # print ('')

    # print ("U shape", U.shape)
    # print ("S shape", S.shape)
    # print ("V shape", V.shape)
    
    # print ("V trnspose", V.transpose().shape)

    result = np.matmul(npUserTermArray, V.transpose())
    print (result)
    print (result.shape)

def pca_reduction(array):
    print ("PCA::::::::")

    covMatrix = np.cov(npUserTermArray)
    print (covMatrix.diagonal())
    # U, singularValues, V = linalg.svd(covMatrix,full_matrices=False)

    # print ("singularValues")
    # print (singularValues)
    # print (singularValues.shape)

    # termWeightMatrix = np.dot(npUserTermArray.transpose(),U)
    # termWeightMatrix = np.matmul(U.transpose(), npUserTermArray)
    # termWeightMatrix = np.matmul(npUserTermArray.transpose(), V.transpose())

    # print ("Term Weight Array")
    # print (termWeightMatrix)

    # print ("Term Weight Array shape")
    # print (termWeightMatrix.shape)

    print ("::::::::::::")

def lda_red():
    sa = lil_matrix(userTermArray)
    ldamodel = gensim.models.ldamodel.LdaModel(sa, num_topics=2)
    topics = ldamodel.get_topics()
    print (topics)
    print (topics.shape)



client = MongoClient('localhost', 27017)
db = client['test_data']

users = db.descUser.find()
user_count = db.descUser.count()
all_terms = db.descUser.distinct("terms.term")

#all_terms = sorted(all_terms)

#print (all_terms)
#print (len(all_terms))

userTermArray = []

for user in users:
    termArr = [d for d in user["terms"]]
    #tfArr = [d["tf"] for d in user["terms"]]
    tempArr= np.full((55180), 0)

    for term in termArr:
        if term['term'] in all_terms:
            index = all_terms.index(term['term'])
            tempArr[index] = term['tf']
    userTermArray.append(tempArr)


npUserTermArray = np.array(userTermArray)

###Numpy SVD
#svd_reduction(npUserTermArray)


### PCA
#pca_reduction(npUserTermArray)

### LDA
lda_red()

print (datetime.now() - startTime)