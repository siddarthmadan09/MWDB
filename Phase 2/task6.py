from pymongo import MongoClient
import sys
from datetime import datetime
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from scipy.linalg import svd
from numpy import linalg
import gensim

startTime = datetime.now()
#create mongo connection
client = MongoClient('localhost', 27017)
db = client['dev_data']
#fetch locations
locations = db.descLocation.find()
locarr = []
#fetch all distinct terms present
all_terms = db.descLocation.distinct("terms.term")
locationTermArray = []

npLTArray = np.array([]);
npLTTransposeArray = np.array([]);
LLSMatrix = []
LFMatrix = []
k = int(sys.argv[1])
#form location term array based on descriptor
def form_matrix(descriptor):
    global npLTArray,locarr
    for location in locations:
        locarr.append(location)
        termArr = location["terms"]
        tempArr= np.full((55180), 0)
        for term in termArr:
            index = all_terms.index(term['term'])
            tempArr[index] = term[descriptor]
        locationTermArray.append(tempArr)
    npLTArray = np.array(locationTermArray)
    
#perform svd and get top k latent symantics and multiply it original similarity matrix to obtain weight matrix
def performSVD():
    global LFMatrix,k
    U, s, VT = svd(LLSMatrix)
    U = U[:,:k]
    LFMatrix = np.matmul(LLSMatrix,U)

def writeoutput(line):
    with open('./output_6.txt', 'a') as output_file:
        output_file.write(line+'\n')
                 
def main():
    global npLTTransposeArray,LLSMatrix,LFMatrix
    form_matrix('tf-idf')
    npLTTransposeArray = npLTArray.transpose()
    # Perform d * d transpose to obtain location location similarity matrix
    LLSMatrix = np.matmul(npLTArray,npLTTransposeArray)
    #perform svd and get weight matrix
    performSVD();
    index = 1
    #transpose the matrix to display the result in form of each latent symantic
    LFMatrixT = LFMatrix.transpose()
    
    for row in LFMatrixT:
        #for each symantic sort location weights and their indices to obtain location names 
        sortarr = np.sort(row)
        sortarg = np.argsort(row)
        count = sortarr.shape
        writeoutput ("------------------------")
        writeoutput ("Latent Symantic "+str(index))
        #display locations for each symatic in deacreasing order of weights
        for i in range(count[0]):
            writeoutput("location:"+locarr[sortarg[count[0]-i-1]]['locationName']+" Weight: "+ str(sortarr[count[0]-i-1]))
        index += 1
        
    
    
main()
print (datetime.now() - startTime)
