from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client['projectmwdb']  # use a database called "test_database"
collection = db['devset_textTermsPerUser']

file = "devset_textTermsPerUser.txt"

f = open(file, encoding='utf8')  # open a file
text = f.readlines()    

text = [x.strip() for x in text]
collection.delete_many({})

for line in text:
    words = line.split(" ")
    userid = words[0]
    arr_terms = []
    for x in range(1, len(words), 4):
        arr_terms.append({
            'Term' : words[x][1:-1],
            'TF' : float(words[x+1]),
            'DF' : float(words[x+2]),
            'TF-IDF' : float(words[x+3])
        })
    # insert the contents into the "file" collection
    collection.insert_one({
        'id' : userid,
        'terms' : arr_terms
    })
