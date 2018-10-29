import sys
from datetime import datetime
import numpy as np
import pandas as pd

import task1_2_base

startTime = datetime.now()

## input: python task1.py user/image/location svd/pca/lda k

### Start
if len(sys.argv) == 4:
    dataType = str(sys.argv[1])
    decompositionMethod = str(sys.argv[2])
    k = int(sys.argv[3])

    if dataType == "image":
        #devsetDirectoryPath = open('devset_directory_path.config', 'r').read()
        dataArray, docs, terms = task1_2_base.computeImageTermArray('./data/devset_textTermsPerImage.csv')
    else:
        dataArray, docs, terms = task1_2_base.computeDataArray(dataFamily=dataType)

    if decompositionMethod == "svd":
        termWeightPairs = task1_2_base.svd_reduction(dataArray, k)
        print (termWeightPairs)
    elif decompositionMethod == "pca":
        termWeightPairs = task1_2_base.pca_reduction(dataArray, k)
        print (termWeightPairs)
    elif decompositionMethod == "lda":
        termWeightPairs = task1_2_base.lda_reduction(dataArray, k)

        # for i, topic_dist in enumerate(termWeightPairs):
        #     topic_words = np.array(terms)[np.argsort(topic_dist)][:-10:-1]
        #     print('Topic {}: {}'.format(i, ' '.join(topic_words)))

        print (termWeightPairs)
    else:
        print ("Invalid decomposition method")
        sys.exit(0)
else:
    print ("Invalid input")
    sys.exit(0)

dataframe = pd.DataFrame(data=termWeightPairs.astype(float))
dataframe.to_csv('outfile_'+ dataType + '_' + decompositionMethod + '_' + str(k) +'.csv', sep=' ', header=False, float_format='%.2f', index=False)

print ("Total time taken: ", datetime.now() - startTime)