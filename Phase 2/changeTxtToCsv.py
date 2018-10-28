import csv

with open('devset_textTermsPerUser.txt') as fin, open('devset_textTermsPerUserNew.csv', 'w') as fout:
    o=csv.writer(fout)
    for line in fin:
        o.writerow(line.split())
