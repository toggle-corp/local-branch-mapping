from numpy.random import multinomial

from numpy import log, exp

from numpy import argmax

import json

import numpy as np

import gensim
class MovieGroupProcess:

    def __init__(self, K=8, alpha=0.1, beta=0.1, n_iters=30):

        '''

        A MovieGroupProcess is a conceptual model introduced by Yin and Wang 2014 to

        describe their Gibbs sampling algorithm for a Dirichlet Mixture Model for the

        clustering short text documents.

        Reference: http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        Imagine a professor is leading a film class. At the start of the class, the students

        are randomly assigned to K tables. Before class begins, the students make lists of

        their favorite films. The teacher reads the role n_iters times. When

        a student is called, the student must select a new table satisfying either:

            1) The new table has more students than the current table.

        OR

            2) The new table has students with similar lists of favorite movies.

        :param K: int

            Upper bound on the number of possible clusters. Typically many fewer

        :param alpha: float between 0 and 1

            Alpha controls the probability that a student will join a table that is currently empty

            When alpha is 0, no one will join an empty table.

        :param beta: float between 0 and 1

            Beta controls the student's affinity for other students with similar interests. A low beta means

            that students desire to sit with students of similar interests. A high beta means they are less

            concerned with affinity and are more influenced by the popularity of a table

        :param n_iters:

        '''

        self.K = K

        self.alpha = alpha

        self.beta = beta

        self.n_iters = n_iters



        # slots for computed variables

        self.number_docs = None

        self.vocab_size = None

        self.cluster_doc_count = [0 for _ in range(K)]

        self.cluster_word_count = [0 for _ in range(K)]

        self.cluster_word_distribution = [{} for i in range(K)]



    @staticmethod

    def from_data(K, alpha, beta, D, vocab_size, cluster_doc_count, cluster_word_count, cluster_word_distribution):

        '''

        Reconstitute a MovieGroupProcess from previously fit data

        :param K:

        :param alpha:

        :param beta:

        :param D:

        :param vocab_size:

        :param cluster_doc_count:

        :param cluster_word_count:

        :param cluster_word_distribution:

        :return:

        '''

        mgp = MovieGroupProcess(K, alpha, beta, n_iters=30)

        mgp.number_docs = D

        mgp.vocab_size = vocab_size

        mgp.cluster_doc_count = cluster_doc_count

        mgp.cluster_word_count = cluster_word_count

        mgp.cluster_word_distribution = cluster_word_distribution

        return mgp



    @staticmethod

    def _sample(p):

        '''

        Sample with probability vector p from a multinomial distribution

        :param p: list

            List of probabilities representing probability vector for the multinomial distribution

        :return: int

            index of randomly selected output

        '''

        return [i for i, entry in enumerate(multinomial(1, p)) if entry != 0][0]



    def fit(self, docs, vocab_size):

        '''

        Cluster the input documents

        :param docs: list of list

            list of lists containing the unique token set of each document

        :param V: total vocabulary size for each document

        :return: list of length len(doc)

            cluster label for each document

        '''

        alpha, beta, K, n_iters, V = self.alpha, self.beta, self.K, self.n_iters, vocab_size



        D = len(docs)

        self.number_docs = D

        self.vocab_size = vocab_size



        # unpack to easy var names

        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution

        cluster_count = K

        d_z = [None for i in range(len(docs))]



        # initialize the clusters

        for i, doc in enumerate(docs):



            # choose a random  initial cluster for the doc

            z = self._sample([1.0 / K for _ in range(K)])

            d_z[i] = z

            m_z[z] += 1

            n_z[z] += len(doc)



            for word in doc:

                #print(word)

                if word not in n_z_w[z]:

                    n_z_w[z][word] = 0

                n_z_w[z][word] += 1

        #print(n_z_w)

        for _iter in range(n_iters):

            total_transfers = 0



            for i, doc in enumerate(docs):



                # remove the doc from it's current cluster

                z_old = d_z[i]



                m_z[z_old] -= 1

                n_z[z_old] -= len(doc)



                for word in doc:

                    n_z_w[z_old][word] -= 1



                    # compact dictionary to save space

                    if n_z_w[z_old][word] == 0:

                        del n_z_w[z_old][word]



                # draw sample from distribution to find new cluster

                p = self.score(doc)

                z_new = self._sample(p)



                # transfer doc to the new cluster

                if z_new != z_old:

                    total_transfers += 1



                d_z[i] = z_new

                m_z[z_new] += 1

                n_z[z_new] += len(doc)



                for word in doc:

                    if word not in n_z_w[z_new]:

                        n_z_w[z_new][word] = 0

                    n_z_w[z_new][word] += 1



            cluster_count_new = sum([1 for v in m_z if v > 0])

            print("In stage %d: transferred %d clusters with %d clusters populated" % (

            _iter, total_transfers, cluster_count_new))

            if total_transfers == 0 and cluster_count_new == cluster_count and _iter>25:

                print("Converged.  Breaking out.")

                break

            self.cluster_count = cluster_count_new

        self.cluster_word_distribution = n_z_w

        return d_z



    def score(self, doc):

        '''

        Score a document

        Implements formula (3) of Yin and Wang 2014.

        http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        :param doc: list[str]: The doc token stream

        :return: list[float]: A length K probability vector where each component represents

                              the probability of the document appearing in a particular cluster

        '''

        alpha, beta, K, V, D = self.alpha, self.beta, self.K, self.vocab_size, self.number_docs

        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution



        p = [0 for _ in range(K)]



        #  We break the formula into the following pieces

        #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)

        #  lN1 = log(m_z[z] + alpha)

        #  lN2 = log(D - 1 + K*alpha)

        #  lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))

        #  lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))



        lD1 = log(D - 1 + K * alpha)

        doc_size = len(doc)

        #print(doc_size)

        for label in range(K):

            lN1 = log(m_z[label] + alpha)

            lN2 = 0

            lD2 = 0

            for word in doc:

                lN2 += log(n_z_w[label].get(word, 0) + beta)

            for j in range(1, doc_size +1):

                lD2 += log(n_z[label] + V * beta + j - 1)

            #print(lD2)

            p[label] = exp(lN1 - lD1 + lN2 - lD2)



        # normalize the probability vector

        #print(p)

        pnorm = sum(p)

        pnorm = pnorm if pnorm>0 else 1

        return [pp/pnorm for pp in p]



    def choose_best_label(self, doc):

        '''

        Choose the highest probability label for the input document

        :param doc: list[str]: The doc token stream

        :return:

        '''

        p = self.score(doc)

        return argmax(p),max(p)

model=MovieGroupProcess(K=10, alpha=0.1, beta=0.01, n_iters=200)
model.fit(processed_docs,2772)
import pandas as pd

import math
from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

import gensim

import numpy as np

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

#np.random.seed(2018)

import nltk

nltk.download('wordnet')

stemmer = PorterStemmer()
def lemmatize_stemming(text):

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):

    result = []

    token_list=[]

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:

            result.append(lemmatize_stemming(token))

            token_list.append(token)

    return result,dict(zip(result,token_list))
def produce_mapping(mapping_list):

    #processed_ser= corpus.fillna("").apply(lambda x: re.sub(r"http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+"," ",x)).map(preprocess)

    #processed_docs=[item[0] for item in processed_ser]

    #mapping_list=[item[1] for item in processed_ser]

    mapping_pairs=pd.concat([pd.DataFrame([(k,v) for k,v in d.items()]) for d in mapping_list])

    mapping_pairs['count']=1

    mapping121=mapping_pairs.groupby(by=[0,1]).count().reset_index().sort_values(by=[0,'count'],ascending=False).groupby(by=0).head(1)

    mapping12many=mapping_pairs.drop(columns=['count']).drop_duplicates()

    return mapping121,mapping12many
MI_fb_df=pd.read_csv("../data/facebook_Malawi.csv",delimiter="|",index_col=0)

MI_tw_df=pd.read_csv("../data/twitter_Malawi.csv",delimiter="|",index_col=0)
processed_ser = MI_fb_df['message'].fillna("").apply(lambda x: re.sub(r"http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+"," ",x)).map(preprocess)

processed_docs=np.array([np.array(item[0]) for item in processed_ser])

mapping_list=[item[1] for item in processed_ser]
class MovieGroupProcess_np:

    def __init__(self, K=8, alpha=0.1, beta=0.1, n_iters=30):

        '''

        A MovieGroupProcess is a conceptual model introduced by Yin and Wang 2014 to

        describe their Gibbs sampling algorithm for a Dirichlet Mixture Model for the

        clustering short text documents.

        Reference: http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        Imagine a professor is leading a film class. At the start of the class, the students

        are randomly assigned to K tables. Before class begins, the students make lists of

        their favorite films. The teacher reads the role n_iters times. When

        a student is called, the student must select a new table satisfying either:

            1) The new table has more students than the current table.

        OR

            2) The new table has students with similar lists of favorite movies.

        :param K: int

            Upper bound on the number of possible clusters. Typically many fewer

        :param alpha: float between 0 and 1

            Alpha controls the probability that a student will join a table that is currently empty

            When alpha is 0, no one will join an empty table.

        :param beta: float between 0 and 1

            Beta controls the student's affinity for other students with similar interests. A low beta means

            that students desire to sit with students of similar interests. A high beta means they are less

            concerned with affinity and are more influenced by the popularity of a table

        :param n_iters:

        '''

        self.K = K

        self.alpha = alpha

        self.beta = beta

        self.n_iters = n_iters



        # slots for computed variables

        self.number_docs = None

        self.vocab_size = None

        self.cluster_doc_count = np.repeat(0,K)

        self.cluster_word_count = np.repeat(0,K)

        self.cluster_word_distribution =None

#         self.cluster_doc_count = [0 for _ in range(K)]

#         self.cluster_word_count = [0 for _ in range(K)]

#         self.cluster_word_distribution = [{} for i in range(K)]



    @staticmethod

    def from_data(K, alpha, beta, D, vocab_size, cluster_doc_count, cluster_word_count, cluster_word_distribution):

        '''

        Reconstitute a MovieGroupProcess from previously fit data

        :param K:

        :param alpha:

        :param beta:

        :param D:

        :param vocab_size:

        :param cluster_doc_count:

        :param cluster_word_count:

        :param cluster_word_distribution:

        :return:

        '''

        mgp = MovieGroupProcess(K, alpha, beta, n_iters=30)

        mgp.number_docs = D

        mgp.vocab_size = vocab_size

        mgp.cluster_doc_count = cluster_doc_count

        mgp.cluster_word_count = cluster_word_count

        mgp.cluster_word_distribution = cluster_word_distribution

        return mgp



    @staticmethod

    def _sample(p):

        '''

        Sample with probability vector p from a multinomial distribution

        :param p: list

            List of probabilities representing probability vector for the multinomial distribution

        :return: int

            index of randomly selected output

        '''

        #return np.where(multinomial(1,p)==1)[0][0]

        return [i for i, entry in enumerate(multinomial(1, p)) if entry != 0][0]



    def fit(self, docs, vocab_size):

        '''

        Cluster the input documents

        :param docs: list of list

            list of lists containing the unique token set of each document

        :param V: total vocabulary size for each document

        :return: list of length len(doc)

            cluster label for each document

        '''

        alpha, beta, K, n_iters, V = self.alpha, self.beta, self.K, self.n_iters, vocab_size



        D = len(docs)

        self.number_docs = D

        self.vocab_size = vocab_size



        # unpack to easy var names

        self.cluster_word_distribution=np.zeros((K,vocab_size))

        print(self.cluster_word_distribution.shape)

        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution

        cluster_count = K

        d_z = np.repeat(0,D)



        # initialize the clusters

        for i, doc in enumerate(docs):



            # choose a random  initial cluster for the doc

            z = self._sample([1.0 / K for _ in range(K)])

#             if sum(doc>0)==0:

#                 continue

            d_z[i] = z

            m_z[z] += 1

            n_z[z] += sum(doc)

            n_z_w[z]+=doc

#             for word in doc:

#                 #print(word)

#                 if word not in n_z_w[z]:

#                     n_z_w[z][word] = 0

#                 n_z_w[z][word] += 1

        #print(n_z_w)

        for _iter in range(n_iters):

            total_transfers = 0



            for i, doc in enumerate(docs):



                # remove the doc from it's current cluster

                z_old = d_z[i]

#                 if sum(doc>0)==0:

#                     continue

                m_z[z_old] -= 1

                n_z[z_old] -= sum(doc)

                n_z_w[z_old]-=doc

#                 for word in doc:

#                     n_z_w[z_old][word] -= 1



#                     # compact dictionary to save space

#                     if n_z_w[z_old][word] == 0:

#                         del n_z_w[z_old][word]



                # draw sample from distribution to find new cluster

                p = self.score(doc)

                z_new = self._sample(p)

                #print(z_new)

                # transfer doc to the new cluster

                if z_new != z_old:

                    total_transfers += 1



                d_z[i] = z_new

                m_z[z_new] += 1

                n_z[z_new] += sum(doc)

                n_z_w[z_new]+=doc

#                 for word in doc:

#                     if word not in n_z_w[z_new]:

#                         n_z_w[z_new][word] = 0

#                     n_z_w[z_new][word] += 1



            cluster_count_new = sum([1 for v in m_z if v > 0])

            print("In stage %d: transferred %d clusters with %d clusters populated" % (

            _iter, total_transfers, cluster_count_new))

            if total_transfers == 0 and cluster_count_new == cluster_count and _iter>25:

                print("Converged.  Breaking out.")

                break

            self.cluster_count = cluster_count_new

        self.cluster_word_distribution = n_z_w

        return d_z



    def score(self, doc):

        '''

        Score a document

        Implements formula (3) of Yin and Wang 2014.

        http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        :param doc: list[str]: The doc token stream

        :return: list[float]: A length K probability vector where each component represents

                              the probability of the document appearing in a particular cluster

        '''

        alpha, beta, K, V, D = self.alpha, self.beta, self.K, self.vocab_size, self.number_docs

        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution



        #p = np.repeat(0,K)



        #  We break the formula into the following pieces

        #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)

        #  lN1 = log(m_z[z] + alpha)

        #  lN2 = log(D - 1 + K*alpha)

        #  lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))

        #  lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))



        lD1 = log(D - 1 + K * alpha)

        doc_size = sum(doc)

        #print(doc_size)

        #print(sum(doc),sum(doc>0))

        lN2=np.dot(np.log(n_z_w+beta),doc).T[0]

        #lN2=doc[doc>0]*np.log(n_z_w[doc>0]+beta)

        

        #print([[np.log(n_z[i]+V * beta + j - 1) for j in range(1, doc_size +1)] for i in range(K)])

        #print(n_z)

        lD2=np.array([np.sum([np.log(n_z[i]+V * beta + j - 1) for j in range(1, doc_size +1)]) for i in range(K)])

        lN1=np.log(m_z+alpha)

        #print(lD2)

        p=np.exp(lN1 - lD1 + lN2 - lD2)

#         for label in range(K):

#             lN1 = log(m_z[label] + alpha)

#             lN2 = 0

#             lD2 = 0

#             for word in doc:

#                 lN2 += log(n_z_w[label].get(word, 0) + beta)

#             for j in range(1, doc_size +1):

#                 lD2 += log(n_z[label] + V * beta + j - 1)

#             p[label] = exp(lN1 - lD1 + lN2 - lD2)



        # normalize the probability vector

        #print(lN1,lD1 , lN2 , lD2)

        #print(p)

        pnorm = sum(p)

        pnorm = pnorm if pnorm>0 else 1

        #return [pp/pnorm for pp in p]

        return p/pnorm

    

    def choose_best_label(self, doc):

        '''

        Choose the highest probability label for the input document

        :param doc: list[str]: The doc token stream

        :return:

        '''

        p = self.score(doc)

        return argmax(p),max(p)

dictionary = gensim.corpora.Dictionary(processed_docs)
from scipy.sparse import csr_matrix
from functools import reduce

word_size=len(set(reduce(lambda x,y:list(x)+list(y),processed_docs)))
vec_docs=[csr_matrix((a[1],(np.repeat(0,len(a[0])),a[0]))).toarray().flatten()   for a in [list(zip(*dictionary.doc2bow(doc))) for doc in processed_docs] if len(a)>0 ]
model=MovieGroupProcess_np(K=10, alpha=0.1, beta=0.01, n_iters=200)
vec_docs=[csr_matrix((a[1],(np.repeat(0,len(a[0])),a[0])),shape=(1,word_size)).toarray().flatten()  for a in [list(zip(*dictionary.doc2bow(doc))) for doc in processed_docs] if len(a)>0]
model.fit(vec_docs,word_size)
class MovieGroupProcess_py:

    def __init__(self, K=8, alpha=0.1, beta=0.1, n_iters=30):

        '''

        A MovieGroupProcess is a conceptual model introduced by Yin and Wang 2014 to

        describe their Gibbs sampling algorithm for a Dirichlet Mixture Model for the

        clustering short text documents.

        Reference: http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        Imagine a professor is leading a film class. At the start of the class, the students

        are randomly assigned to K tables. Before class begins, the students make lists of

        their favorite films. The teacher reads the role n_iters times. When

        a student is called, the student must select a new table satisfying either:

            1) The new table has more students than the current table.

        OR

            2) The new table has students with similar lists of favorite movies.

        :param K: int

            Upper bound on the number of possible clusters. Typically many fewer

        :param alpha: float between 0 and 1

            Alpha controls the probability that a student will join a table that is currently empty

            When alpha is 0, no one will join an empty table.

        :param beta: float between 0 and 1

            Beta controls the student's affinity for other students with similar interests. A low beta means

            that students desire to sit with students of similar interests. A high beta means they are less

            concerned with affinity and are more influenced by the popularity of a table

        :param n_iters:

        '''

        self.K = K

        self.alpha = alpha

        self.beta = beta

        self.n_iters = n_iters



        # slots for computed variables

        self.number_docs = None

        self.vocab_size = None

        self.cluster_doc_count = np.repeat(0,K)

        self.cluster_word_count = np.repeat(0,K)

        self.cluster_word_distribution = [{} for i in range(K)]



    @staticmethod

    def from_data(K, alpha, beta, D, vocab_size, cluster_doc_count, cluster_word_count, cluster_word_distribution):

        '''

        Reconstitute a MovieGroupProcess from previously fit data

        :param K:

        :param alpha:

        :param beta:

        :param D:

        :param vocab_size:

        :param cluster_doc_count:

        :param cluster_word_count:

        :param cluster_word_distribution:

        :return:

        '''

        mgp = MovieGroupProcess(K, alpha, beta, n_iters=30)

        mgp.number_docs = D

        mgp.vocab_size = vocab_size

        mgp.cluster_doc_count = cluster_doc_count

        mgp.cluster_word_count = cluster_word_count

        mgp.cluster_word_distribution = cluster_word_distribution

        return mgp



    @staticmethod

    def _sample(p):

        '''

        Sample with probability vector p from a multinomial distribution

        :param p: list

            List of probabilities representing probability vector for the multinomial distribution

        :return: int

            index of randomly selected output

        '''

        return [i for i, entry in enumerate(multinomial(1, p)) if entry != 0][0]



    def fit(self, docs, vocab_size):

        '''

        Cluster the input documents

        :param docs: list of list

            list of lists containing the unique token set of each document

        :param V: total vocabulary size for each document

        :return: list of length len(doc)

            cluster label for each document

        '''

        alpha, beta, K, n_iters, V = self.alpha, self.beta, self.K, self.n_iters, vocab_size



        D = len(docs)

        self.number_docs = D

        self.vocab_size = vocab_size



        # unpack to easy var names

        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution

        cluster_count = K

        d_z = [None for i in range(len(docs))]



        # initialize the clusters

        for i, doc in enumerate(docs):



            # choose a random  initial cluster for the doc

            z = self._sample([1.0 / K for _ in range(K)])

            d_z[i] = z

            m_z[z] += 1

            n_z[z] += len(doc)



            for word in doc:

                #print(word)

                if word not in n_z_w[z]:

                    n_z_w[z][word] = 0

                n_z_w[z][word] += 1

        #print(n_z_w)

        for _iter in range(n_iters):

            total_transfers = 0



            for i, doc in enumerate(docs):



                # remove the doc from it's current cluster

                z_old = d_z[i]



                m_z[z_old] -= 1

                n_z[z_old] -= len(doc)



                for word in doc:

                    n_z_w[z_old][word] -= 1



                    # compact dictionary to save space

                    if n_z_w[z_old][word] == 0:

                        del n_z_w[z_old][word]



                # draw sample from distribution to find new cluster

                p = self.score(doc)

                z_new = self._sample(p)



                # transfer doc to the new cluster

                if z_new != z_old:

                    total_transfers += 1



                d_z[i] = z_new

                m_z[z_new] += 1

                n_z[z_new] += len(doc)



                for word in doc:

                    if word not in n_z_w[z_new]:

                        n_z_w[z_new][word] = 0

                    n_z_w[z_new][word] += 1



            cluster_count_new = sum([1 for v in m_z if v > 0])

            print("In stage %d: transferred %d clusters with %d clusters populated" % (

            _iter, total_transfers, cluster_count_new))

            if total_transfers == 0 and cluster_count_new == cluster_count and _iter>25:

                print("Converged.  Breaking out.")

                break

            self.cluster_count = cluster_count_new

        self.cluster_word_distribution = n_z_w

        return d_z



    def score(self, doc):

        '''

        Score a document

        Implements formula (3) of Yin and Wang 2014.

        http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        :param doc: list[str]: The doc token stream

        :return: list[float]: A length K probability vector where each component represents

                              the probability of the document appearing in a particular cluster

        '''

        alpha, beta, K, V, D = self.alpha, self.beta, self.K, self.vocab_size, self.number_docs

        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution



        p = [0 for _ in range(K)]



        #  We break the formula into the following pieces

        #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)

        #  lN1 = log(m_z[z] + alpha)

        #  lN2 = log(D - 1 + K*alpha)

        #  lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))

        #  lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))

        

        lD1 = log(D - 1 + K * alpha)

        doc_size = len(doc)

        #print(doc_size)

        p=[exp(log(m_z[label] + alpha)-lD1+sum([log(n_z_w[label].get(word, 0) + beta) for word in doc])-sum([log(n_z[label] + V * beta + j - 1) for j in range(1, doc_size +1)])) for label in range(K)]

#         for label in range(K):

#             lN1 = log(m_z[label] + alpha)

#             lN2 = 0

#             lD2 = 0

#             for word in doc:

#                 lN2 += log(n_z_w[label].get(word, 0) + beta)

#             for j in range(1, doc_size +1):

#                 lD2 += log(n_z[label] + V * beta + j - 1)

#             #print(lD2)

#             p[label] = exp(lN1 - lD1 + lN2 - lD2)



        # normalize the probability vector

        #print(p)

        pnorm = sum(p)

        pnorm = pnorm if pnorm>0 else 1

        return [pp/pnorm for pp in p]



    def choose_best_label(self, doc):

        '''

        Choose the highest probability label for the input document

        :param doc: list[str]: The doc token stream

        :return:

        '''

        p = self.score(doc)

        return argmax(p),max(p)

model=MovieGroupProcess_py(K=10, alpha=0.1, beta=0.01, n_iters=200)
%time model.fit(processed_docs,2772)
model=MovieGroupProcess(K=10, alpha=0.1, beta=0.01, n_iters=200)
%time model.fit(processed_docs,2772)