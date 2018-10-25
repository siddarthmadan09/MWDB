from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client['projectmwdb']  # use a database called "test_database"
collection = db['devset_textTermsPerPOI']

file = "devset_textTermsPerPOI.txt"

f = open(file, encoding='utf8')  # open a file
text = f.readlines()    

text = [x.strip() for x in text]
collection.delete_many({})

for line in text:
    words = line.split(' ')
    i=0
    locid = ''
    while(words[i][0] != '"'):
        locid += words[i]+' '
        i += 1
    locid = locid[:-1]
    arr_terms = []
    for x in range(i, len(words), 4):
        arr_terms.append({
            'Term' : words[x][1:-1],
            'TF' : float(words[x+1]),
            'DF' : float(words[x+2]),
            'TF-IDF' : float(words[x+3])
        })
    # insert the contents into the "file" collection
    collection.insert_one({
        'id' : locid,
        'terms' : arr_terms
    })
