from pymongo import MongoClient

def addToDB(document, fileData):
    allData = []
    record = {}
    keyName = document + 'Id'
    for user in fileData:
        terms = []

        keyValue = user.partition('"')[0]
        user = user.replace(keyValue, '')
        if document == 'location':
            keyName = 'locationName'
            keyValue = keyValue.strip().replace(' ', '_')

        oneline = user.split(" ")

        n,m = 0,0
        while m + 4 < len(oneline): 
            n = m
            m = n + 4
            temp = oneline[n:m]

            temp[0] = temp[0].replace('"', '')
            record = {
                'term': temp[0],
                'tf': int(temp[1]),
                'df': int(temp[2]),
                'tf-idf': float(temp[3])
            }
            terms.append(record)

        currentData = {
            keyName: keyValue.strip(),
            "terms": terms
        }

        allData.append(currentData)
    return allData

client = MongoClient()
db = client['dev_data']
devsetDirectoryPath = open('devset_directory_path.config', 'r').read()

filename = "/devset/desctxt/devset_textTermsPerUser.txt"
fileData = open(devsetDirectoryPath + filename, "r").read()
fileData = fileData.split('\n')
dataToSave = addToDB('user', fileData)
db.descUser.insert_many(dataToSave)

filename = "/devset/desctxt/devset_textTermsPerImage.txt"
fileData = open(devsetDirectoryPath + filename, "r").read()
fileData = fileData.split('\n')
dataToSave = addToDB('image', fileData)
db.descImage.insert_many(dataToSave)

filename = "/devset/desctxt/devset_textTermsPerPOI.txt"
fileData = open(devsetDirectoryPath + filename, "r").read()
fileData = fileData.split('\n')
dataToSave = addToDB('location', fileData)
db.descLocation.insert_many(dataToSave)