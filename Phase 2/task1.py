import sys
from pymongo import MongoClient
from datetime import datetime
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from numpy import linalg
import gensim, lda

startTime = datetime.now()

def svd_reduction(dataArray, k):
    U, S, V = linalg.svd(dataArray, full_matrices=False)

    termWeightMatrix = np.matmul(dataArray.transpose(), U[:,:k])
    
    print ("Term Weight Array")
    print (termWeightMatrix)

    print ("Term Weight Array shape")
    print (termWeightMatrix.shape)


def pca_reduction(dataArray, k):
    covMatrix = np.cov(dataArray)
    U, singularValues, V = linalg.svd(covMatrix,full_matrices=False)

    termWeightMatrix = np.dot(dataArray.transpose(), U[:,:k])

    print ("Term Weight Array")
    print (termWeightMatrix)

    print ("Term Weight Array shape")
    print (termWeightMatrix.shape)

def lda_reduction(dataArray, k):
        sparseDataArray = lil_matrix(dataArray)

        model = lda.LDA(n_topics=20)
        model.fit(sparseDataArray)  # model.fit_transform(X) is also available
        topic_word = model.topic_word_  # model.components_ also works
        n_top_words = 8
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(all_terms)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))

        doc_topic = model.doc_topic_
        for i in range(10):
            print("{} (top topic: {})".format(all_users[i], doc_topic[i].argmax()))


client = MongoClient('localhost', 27017)
db = client['test_data']

users = db.descUser.find()
user_count = db.descUser.count()
all_terms = db.descUser.distinct("terms.term")

userTermArray = []
all_users = []

for user in users:
    termArr = [d for d in user["terms"]]
    all_users.append(user["userId"])
    #tfArr = [d["tf"] for d in user["terms"]]
    tempArr= np.full((55180), 0)

    for term in termArr:
        if term['term'] in all_terms:
            index = all_terms.index(term['term'])
            tempArr[index] = term['tf']
    userTermArray.append(tempArr)


npUserTermArray = np.array(userTermArray)

###Numpy SVD
#svd_reduction(npUserTermArray, k)


### PCA
#pca_reduction(npUserTermArray, 8)

### LDA
lda_reduction(npUserTermArray, 8)

print (datetime.now() - startTime)