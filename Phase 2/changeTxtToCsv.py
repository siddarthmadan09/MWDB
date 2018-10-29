import csv

with open('./data/devset_textTermsPerUser.txt') as fin, open('./data/devset_textTermsPerUser.csv', 'w') as fout:
    o=csv.writer(fout)
    for line in fin:
        o.writerow(line.split())

with open('./data/devset_textTermsPerImage.txt') as fin, open('./data/devset_textTermsPerImage.csv', 'w') as fout:
    o=csv.writer(fout)
    for line in fin:
        o.writerow(line.split())

with open('./data/devset_textTermsPerPOI.wFolderNames.txt') as fin, open('./data/devset_textTermsPerPOI.wFolderNames.csv', 'w') as fout:
    o=csv.writer(fout)
    for line in fin:
        o.writerow(line.split())

