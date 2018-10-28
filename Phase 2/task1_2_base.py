import sys
from pymongo import MongoClient
from datetime import datetime
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from numpy import linalg
import gensim, lda
from sparsesvd import sparsesvd

def computeDataArray(dataFamily="user"):
    client = MongoClient('localhost', 27017)
    db = client['dev_data']
    collection = db.descUser

    if dataFamily == "user":
        collection = db.descUser
    elif dataFamily == "image":
        collection = db.descImage
    else:
        ##convert location id to location name first
        collection = db.descLocation

    all_data = collection.find()

    all_terms = collection.distinct("terms.term")

    documentTermArray = []
    
    key = dataFamily + "Id"
    if dataFamily == "location":
        key = "locationName"
    
    all_documents = collection.distinct(key)

    for data in all_data:
        all_documents.append(data[key])
        #tfArr = [d["tf"] for d in user["terms"]]
        tempArr= np.full((len(all_terms)), 0, dtype=np.int8)

        for term in data["terms"]:
            if term['term'] in all_terms:
                index = all_terms.index(term['term'])
                tempArr[index] = term['tf']
        documentTermArray.append(tempArr)

    npDocumentTermArray = np.array(documentTermArray, dtype=np.int8)

    return npDocumentTermArray, all_documents, all_terms


def svd_reduction(dataArray, k, get="feature-latent"):
    sparseDataArray = csc_matrix(dataArray)
    ut, s, vt = sparsesvd(sparseDataArray, k)

    if get=="feature-latent":
        return np.matmul(dataArray.transpose(), ut.transpose())
    else:
        return np.matmul(dataArray, vt.transpose())


def pca_reduction(dataArray, k, get="feature-latent"):
    covMatrix = np.cov(dataArray)
    sparseDataArray = csc_matrix(covMatrix)
    ut, s, vt = sparsesvd(sparseDataArray, k)

    if get=="feature-latent":
        return np.matmul(dataArray.transpose(), ut.transpose())
    else:
        return np.matmul(dataArray, vt.transpose())



def lda_reduction(dataArray, k):
    sparseDataArray = lil_matrix(dataArray)

    model = lda.LDA(n_topics=k, n_iter=200)
    model.fit(sparseDataArray)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    doc_topic = model.doc_topic_

    return topic_word, doc_topic

    
def euclideansimilarity(objlatentpairs,docs,terms,dataId):
    index_original = docs.index(dataId)
    # print ("Index of dataId: ", index_original)
    temp_arr_ls_docs = objlatentpairs[index_original]
    # print ("First vector in object latent paurs: ", objlatentpairs[:1,:])
    # print ("Shape of data vector: ", temp_arr_ls_docs.shape)
    ## iterate thru each row, subtract with data, get norm - > similarity score for that row 
    resultArray = []
    for i in range(0, len(objlatentpairs)):
        sub = np.subtract(objlatentpairs[i], temp_arr_ls_docs)
        dist = np.linalg.norm(sub)
        resultArray.append({
            "dataId": docs[i],
            "score": dist
        })

    resultArray = sorted(resultArray, key=lambda k: k['score'])
    print ("============ Most Similar 5 Data Ids ==============")

    for i in range(5):
        print (i+1, "DataId:", resultArray[i]['dataId'], "Score = ", resultArray[i]['score'] )

    # print ("Shape of subtracted: ", sub.shape)
    
    # print ("shape of L2 distance", dist.shape)
    # print ("First vector in L2 dist: ", dist)