import random
import math
from collections import defaultdict, namedtuple


from scipy.special import psi as digamma



class LDA(object):
    wordTopicDistribution = namedtuple("wordTopicDistribution", ["wordCount", "topicDistribution"])

    def __init__(self, K, eta, alpha, bagOfWords, count_vect, seed=20):

        self.bagOfWords = bagOfWords
        self.K = K
        self.eta = eta
        self.seed = seed
        self.lamda = []  # lamda[k][anyWord]
        self.phi = defaultdict(lambda : LDA.wordTopicDistribution(0, [0]*K))   # (document_number, term_number) => (wordCount, topicDistribution)
        for docNum, rown in enumerate(bagOfWords):
            for i, d in zip(rown.indices, rown.data):
                self.phi[(docNum, i)] = LDA.wordTopicDistribution(d, [0.001]*3)

        self.alpha = alpha
        self.gamma = [[1.0/K for topic in range(K)] for doc in bagOfWords]
        self.count_vect = count_vect
        self.indexToWord = dict(zip(count_vect.vocabulary_.values(), count_vect.vocabulary_.keys()))



    def initializeLamda(self):
        for k in range(self.K):

            temp = [self.eta for i in range(len(self.count_vect.vocabulary_))]

            total = 0.0
            for docNumber in random.sample(range(self.bagOfWords.shape[0]), self.seed):
                doc = self.bagOfWords[docNumber]
                for wordIndex, wordCount in zip(doc.indices, doc.data):
                    temp[wordIndex] += wordCount
                    total += wordCount

            self.lamda.append(temp[:])



    def updateLambda(self, k, v):

        self.lamda[k][v] = self.eta + \
            sum(
                self.phi[(docNum, v)].topicDistribution[k] *\
                self.phi[(docNum, v)].wordCount
                    for docNum, doc in enumerate(self.bagOfWords))



    def updateGamma(self, doc):
        for k in range(self.K):
            self.gamma[doc][k] = self.alpha[k] + sum(
                    self.phi[(doc, word)].topicDistribution[k] *\
                    self.phi[(doc, word)].wordCount\
                for word in self.bagOfWords[doc].indices
            )



    def updatePhi(self, doc):
        for k in range(self.K):
            lastDigammaTerm = digamma(sum(self.lamda[k]))
            firstDigammaTerm = digamma(self.gamma[doc][k])
            for word in self.bagOfWords[doc].indices:
                self.phi[(doc, word)].topicDistribution[k] = math.e**(firstDigammaTerm + digamma(self.lamda[k][word]) - lastDigammaTerm)

        for word in self.bagOfWords[doc].indices:
            normFactor = sum(self.phi[(doc, word)].topicDistribution)
            for k in range(self.K):
                self.phi[(doc, word)].topicDistribution[k] /= normFactor



    def meanFieldIteration(self):
        for k in range(self.K):
            for word in self.count_vect.vocabulary_.values():
                self.updateLambda(k,word)

        for doc in range(self.bagOfWords.shape[0]):
            self.updateGamma(doc)
            self.updatePhi(doc)

    def run(self, iterationCount = 2):

        self.initializeLamda()

        for doc in range(self.bagOfWords.shape[0]):
            self.updatePhi(doc)

        for iteration in range(iterationCount):
            self.meanFieldIteration()


    def getBetaTopN(self, n):
        return [[ self.indexToWord[index] for index, score in
                sorted(list(enumerate(k)), key= lambda x:x[1], reverse=True)[:n]] for k in self.lamda]






