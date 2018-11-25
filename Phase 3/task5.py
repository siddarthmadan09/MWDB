import random
from collections import defaultdict
from operator import itemgetter
import numpy as np
import pickle
class LSHImpl:

    def __init__(self,d,k,l,hashfamily):
        self.d = d
        self.k = k
        self.l = l
        self.hashfamily = hashfamily
        self.buckets = []
        self.preComputed = {}
        self.build()

    def build(self):
        hashFuncs = [[self.hashfamily.makeHashFunc() for j in range(self.k)] for i in range(self.l)]
        self.buckets.extend([(hashFunc,defaultdict(lambda:[])) for hashFunc in hashFuncs])
        # for hashFunc in hashFuncs:
        #     for h in hashFunc:
        #         print("offset" , h.offset)
        #         print('-' * 50)

    def reindex(self,points,ids):
        self.indexPoints(points,ids)
        return self.buckets

    def indexPoints(self,points,ids):
        for hashFuncs, bucket in self.buckets:
            for i, point in enumerate(points):
                    bucket[self.computeHash(hashFuncs,point)].append(ids[i])
                # else:
                #     computedHash = self.computeHash1(hashFuncs,point)
                #     bucket[computedHash].append(ids[i])

    def computeHash(self,hashFuncs,point):
        return self.hashfamily.combineHashes([h.computeHash(point) for h in hashFuncs])

    def computeHash1(self,hashFuncs,point):
        return self.hashfamily.combineHashes([(h.computeHash1(point)) for h in hashFuncs])

    def query(self, q, sv, allImageIDs, buckets, t):
        candidates = set()
        queriedHashes = []
        for g,table in buckets:
            print('bucket ength ', len(table))
            # for hashFunc in g:
            #         print("offet" , hashFunc.offset)
            # print('-' * 50)
            queriedHash = self.computeHash(g,q)
            matches = table.get(self.computeHash(g,q),[])
            candidates.update(matches)
            queriedHashes.append(queriedHash)
            print('queried   ' , candidates)
        candidates = [(ix,EuclideanHash.calculateSimilarity(q,sv[getImageIndex(allImageIDs,ix)])) for ix in candidates]
        candidates.sort(key=itemgetter(1))
        # for g,table in buckets:
        #     match = self.requery(q,buckets,queriedHash,t-len(candidates))
        #     candidates.update(match)
        return [queriedHashes, candidates[:t]]

    def requery(self, q, sv ,allImageIDs, buckets, queriedHashes, remaining):
        remainingCandidates = set()
        flag = 0
        count = 0
        counter = 0
        for g,table, in buckets:
            queriedHash = queriedHashes[count]
            count = count + 1
            for k,v in table.items():
                if queriedHash != k and flag == 0:
                    continue
                else:
                    flag = 1
                    if counter ==0 :
                        counter += 1
                        continue
                    matches = table.get(k)
                    remaining = remaining - len(matches)
                    remainingCandidates.update(matches)
                    if remaining <= 0:
                        break
            remainingCandidates = [(ix,EuclideanHash.calculateSimilarity(sv[getImageIndex(allImageIDs,q)], sv[getImageIndex(allImageIDs,ix)])) for ix in remainingCandidates]
            remainingCandidates.sort(key=itemgetter(1))
            if(remaining <= 0):
                break
        return remainingCandidates

class EuclideanFamily:

    def __init__(self, l , k ,d):
        self.l = l
        self.k = k
        self.d = d

    def random_vector(self):
        return [random.gauss(0,1) for i in range(self.d)]

    def offset(self,w):
        return random.uniform(0,w)

    def makeHashFunc(self):
        w=1
        return EuclideanHash(self.random_vector(), self.offset(w), w)

    def dotproduct(u,v):
        return sum(ux*vx for ux,vx in zip(u,v))

    def combineHashes(self,hashes):
        return str(hashes)

class EuclideanHash:

    def __init__(self, random_vec, offset, w):
        self.random_vec = random_vec
        self.offset = offset
        self.w = w

    def computeHash(self,vec):
        return int((self.dotproduct(self.random_vec, vec) + self.offset)/self.w)

    def computeHash1(self,vec):
        return int((self.dotproduct(self.random_vec, vec) + self.offset)/self.w)

    def dotproduct(self,u,v):
        return sum(ux*vx for ux,vx in zip(u,v))

    def calculateSimilarity(x,y):
        return sum((ab - cd)**2 for ab,cd in zip(x,y))**1/2

def getImageIndex(allImageIDs,q):
    for i, imageID in enumerate(allImageIDs):
        if imageID in q:
            index  = i
            break
    return index

# if __name__ == "__main__":
#
#     l = 2
#     k = 2
#     n = 1000
#     d = 5
#     hashFamily = EuclideanFamily(l,k,d)
#
#     lsh = LSHImpl(d,k,l,hashFamily)
#
# #     q = vectors[:int(n/10)]
# # for query in q:
# #     lsh.query(query,3)
#
#     pickle_out = open("lsh.pickle","wb")
#     pickle.dump(lsh, pickle_out)
#     pickle_out.close()



