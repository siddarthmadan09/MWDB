import sys, csv
from pymongo import MongoClient
from datetime import datetime
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from numpy import linalg
import gensim, lda
from sparsesvd import sparsesvd
import pandas as pd
from scipy import spatial

def returnUserTerms(row):
    col_count = 1
    tempTermArray = []
    while col_count < len(row):
        tempTermArray.append(row[col_count])
        col_count += 4
    tempTermArray.pop()
    return tempTermArray

def returnUserTfValues(row):
    col_count = 2
    tempTermArray = []
    while col_count < len(row):
        tempTermArray.append(row[col_count])
        col_count += 4
    return tempTermArray


def getCSVDataAsListData(fileName):
    mainData = []
    with open(fileName) as csv_file:
        csvData = csv.reader(csv_file, delimiter=',')
        for row in csvData:
            mainData.append(row)
        return mainData

def computeImageTermArray(fileName):
    client = MongoClient('localhost', 27017)
    db = client['dev_data']
    collection = db.descUser
    AllTerms = collection.distinct("terms.term")
    AllImages = collection.distinct("imageId")
    UserTermArr = []

    data = getCSVDataAsListData(fileName)
    for row in data:
        termArr = returnUserTerms(row)
        tfArr = returnUserTfValues(row)
        tempArr=[]
        for index in range(len(AllTerms)):
            if AllTerms[index] in termArr:
                loc = termArr.index(AllTerms[index])
                tempArr.append((int)(tfArr[loc]))
            else:
                tempArr.append(0)
        UserTermArr.append(tempArr)
    return np.asarray(UserTermArr), AllImages, AllTerms

def computeDataArray(dataFamily="user"):
    client = MongoClient('localhost', 27017)
    db = client['dev_data']
    collection = db.descUser

    if dataFamily == "user":
        collection = db.descUser
    elif dataFamily == "image":
        collection = db.descImage
    else:
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
        tempArr= np.full((len(all_terms)), 0)

        for term in data["terms"]:
            if term['term'] in all_terms:
                index = all_terms.index(term['term'])
                tempArr[index] = term['tf']
        documentTermArray.append(tempArr)

    npDocumentTermArray = np.array(documentTermArray)

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
    # df = pd.DataFrame(dataArray)
    # covMatrix = df.cov()
    sparseDataArray = csc_matrix(covMatrix)
    ut, s, vt = sparsesvd(sparseDataArray, k)

    if get=="feature-latent":
        return np.matmul(dataArray.transpose(), vt.transpose())
    else:
        return np.matmul(covMatrix, ut.transpose())


def lda_reduction(dataArray, k, get="feature-latent"):
    sparseDataArray = lil_matrix(dataArray)

    model = lda.LDA(n_topics=k, n_iter=200)
    model.fit(sparseDataArray)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    doc_topic = model.doc_topic_
    # print ("topic_word", topic_word.shape)
    # print ("dpc_topic", doc_topic.shape)

    if get == "feature-latent":
        return topic_word
    else:
        return doc_topic


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

    for i in range(10):
        print (i+1, resultArray[i]['dataId'], "| Score = ", resultArray[i]['score'] )

def calculateSimilarityScoreUsingCosine(objlatentpairs, docs, dataId):
    index_original = docs.index(dataId)
    temp_arr_ls_docs = objlatentpairs[index_original] 
    resultArray = []
    for i in range(0, len(objlatentpairs)):
        result = 1 - spatial.distance.cosine(objlatentpairs[i], temp_arr_ls_docs)
        resultArray.append({
            "dataId": docs[i],
            "score": result
        })
    
    resultArray = sorted(resultArray, key=lambda k: k['score'])

    print ("============ Most Similar 5 Data Ids ==============")

    for i in range(10):
        print (i+1, resultArray[i]['dataId'], "| Score = ", resultArray[i]['score'] )

def calculateSimilarityScoreUsingL1(objlatentpairs, docs, dataId):
    index_original = docs.index(dataId)
    temp_arr_ls_docs = objlatentpairs[index_original] 
    resultArray = []
    for i in range(0, len(objlatentpairs)):
        score = 0.0
        for index in range(len(temp_arr_ls_docs)):
            score+=abs(temp_arr_ls_docs[index] - objlatentpairs[i][index])
        
        resultArray.append({
            "dataId": docs[i],
            "score": score
        })
    
    resultArray = sorted(resultArray, key=lambda k: k['score'])

    print ("============ Most Similar 5 Data Ids ==============")

    for i in range(10):
        print (i+1, resultArray[i]['dataId'], "| Score = ", resultArray[i]['score'] )