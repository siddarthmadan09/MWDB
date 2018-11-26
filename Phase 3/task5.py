import random
from collections import defaultdict
from operator import itemgetter


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
        """builds the index structure with the corresponding hash functions and buckets for each layer"""
        hashFuncs = [[self.hashfamily.makeHashFunc() for j in range(self.k)] for i in range(self.l)]
        self.buckets.extend([(hashFunc,defaultdict(lambda:[])) for hashFunc in hashFuncs])

    def reindex(self,points,ids):
        """indexes all the input vectors into the in memory index structure"""
        self.indexPoints(points,ids)
        return self.buckets

    def indexPoints(self,points,ids):
        """indexes the image id of the input vectors into the corresponding buckets for each layer"""
        for hashFuncs, bucket in self.buckets:
            for i, point in enumerate(points):
                    bucket[self.computeHash(hashFuncs,point)].append(ids[i])
                # else:
                #     computedHash = self.computeHash1(hashFuncs,point)
                #     bucket[computedHash].append(ids[i])

    def computeHash(self,hashFuncs,point):
        """compute and combine the hashes of all the hash functions generated"""
        return self.hashfamily.combineHashes([h.computeHash(point) for h in hashFuncs])

    def query(self, q, sv, allImageIDs, buckets, t):
        """compute the hash of the query point to find other similar input points that fall in the same bucket"""
        candidates = set()
        queriedHashes = []
        for g,table in buckets:

            queriedHash = self.computeHash(g,q)
            matches = table.get(self.computeHash(g,q),[])
            candidates.update(matches)
            queriedHashes.append(queriedHash)
        candidates = [(ix,EuclideanHash.calculateSimilarity(q,sv[getImageIndex(allImageIDs,ix)])) for ix in candidates]
        candidates.sort(key=itemgetter(1))

        return [queriedHashes, candidates[:t]]

    def requery(self, q, sv ,allImageIDs, buckets, queriedHashes, remaining):
        """compute the hash of the query point to find other similar input points that fall in neighboring buckets"""
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
        """hash function computed using random projection vectors with offset of bucket size"""
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
    """get image id for the specified index"""
    for i, imageID in enumerate(allImageIDs):
        if imageID in q:
            index  = i
            break
    return index




