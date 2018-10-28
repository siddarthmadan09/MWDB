import sys, xml
from datetime import datetime
import task1_2_base

startTime = datetime.now()

## input format: python task1 user/image/location svd/pca/lda k user/image/location_ID

def getLocationNameFromId(locationId):
    devsetDirectoryPath = open('devset_directory_path.config', 'r').read()
    xmlData = xml.etree.ElementTree.parse(devsetDirectoryPath + '/devset/devset_topics.xml').getroot()

    #get location name from location ID
    for topic in xmlData.findall('topic'):
        if topic.find('number').text == locationId:
            return topic.find('title').text

### Start
if len(sys.argv) == 5:
    dataType = str(sys.argv[1])
    decompositionMethod = str(sys.argv[2])
    k = int(sys.argv[3])
    dataId = str(sys.argv[4])

    if dataType == "location":
        dataId = getLocationNameFromId(dataId)

    dataArray, docs, terms = task1_2_base.computeDataArray(dataFamily=dataType)

    if decompositionMethod == "svd":
        objectLatentPairs = task1_2_base.svd_reduction(dataArray, k, "object-latent")
        task1_2_base.euclideansimilarity(objectLatentPairs, docs, terms, dataId)

    elif decompositionMethod == "pca":
        objectLatentPairs = task1_2_base.pca_reduction(dataArray, k, "object-latent")
        task1_2_base.euclideansimilarity(objectLatentPairs, docs, terms, dataId)

    elif decompositionMethod == "lda":
        documentTopic = task1_2_base.lda_reduction(dataArray, k, "object-latent")
        #task1_2_base.euclideansimilarity(documentTopic, docs, terms, dataId)
        #task1_2_base.calculateSimilarityScoreUsingCosine(documentTopic, docs, dataId)
        task1_2_base.calculateSimilarityScoreUsingL1(documentTopic, docs, dataId)
        # print (documentTopic[0])
        # for i in range(10):
        #     print("{} (top topic: {})".format(docs[i], documentTopic[i].argmax()))

    else:
        print ("Invalid decomposition method")
        sys.exit(0)
else:
    print ("Invalid input")
    sys.exit(0)

print ("Total time taken: ", datetime.now() - startTime)