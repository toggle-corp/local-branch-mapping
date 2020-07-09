from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation

import pandas as pd

pd.set_option('display.expand_frame_repr', False)

import re

import math

import gensim

import numpy as np

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

np.random.seed(2018)

import nltk

nltk.download('wordnet')

stemmer = SnowballStemmer("dutch")

from gensim.models.coherencemodel import CoherenceModel

from functools import reduce

import operator
stopword_list = nltk.corpus.stopwords.words('dutch')

stopword_list += ['httpst', 'httpt', "aan","aangaande","aangezien","achte","achter", \

                  "achterna","af","afgelopen","al","aldaar","aldus","alhoewel","alias", \

                  "alle","allebei","alleen","alles","als","alsnog","altijd","altoos", \

                  "ander","andere","anders","anderszins","beetje","behalve","behoudens", \

                  "beide","beiden","ben","beneden","bent","bepaald","betreffende","bij", \

                  "bijna","bijv","binnen","binnenin","blijkbaar","blijken","boven","bovenal", \

                  "bovendien","bovengenoemd","bovenstaand","bovenvermeld","buiten","bv", \

                  "daar","daardoor","daarheen","daarin","daarna","daarnet","daarom", \

                  "daarop","daaruit","daarvanlangs","dan","dat","de","deden","deed", \

                  "der","derde","derhalve","dertig","deze","dhr","die","dikwijls","dit", \

                  "doch","doe","doen","doet","door","doorgaand","drie","duizend","dus", \

                  "echter","een","eens","eer","eerdat","eerder","eerlang","eerst","eerste", \

                  "eigen","eigenlijk","elk","elke","en","enig","enige","enigszins","enkel", \

                  "er","erdoor","erg","ergens","etc","etcetera","even","eveneens","evenwel", \

                  "gauw","ge","gedurende","geen","gehad","gekund","geleden","gelijk","gemoeten", \

                  "gemogen","genoeg","geweest","gewoon","gewoonweg","haar","haarzelf","had", \

                  "hadden","hare","heb","hebben","hebt","hedden","heeft","heel","hem","hemzelf", \

                  "hen","het","hetzelfde","hier","hierbeneden","hierboven","hierin","hierna", \

                  "hierom","hij","hijzelf","hoe","hoewel","honderd","hun","hunne","ieder","iedere", \

                  "iedereen","iemand","iets","ik","ikzelf","in","inderdaad","inmiddels","intussen", \

                  "inzake","is","ja","je","jezelf","jij","jijzelf","jou","jouw","jouwe","juist", \

                  "jullie","kan","klaar","kon","konden","krachtens","kun","kunnen","kunt","laatst", \

                  "later","liever","lijken","lijkt","maak","maakt","maakte","maakten","maar","mag", \

                  "maken","me","meer","meest","meestal","men","met","mevr","mezelf","mij","mijn", \

                  "mijnent","mijner","mijzelf","minder","miss","misschien","missen","mits","mocht", \

                  "mochten","moest","moesten","moet","moeten","mogen","mr","mrs","mw","na","naar", \

                  "nadat","nam","namelijk","nee","neem","negen","nemen","nergens","net","niemand", \

                  "niet","niets","niks","noch","nochtans","nog","nogal","nooit","nu","nv","of", \

                  "ofschoon","om","omdat","omhoog","omlaag","omstreeks","omtrent","omver","ondanks", \

                  "onder","ondertussen","ongeveer","ons","onszelf","onze","onzeker","ooit","ook","op", \

                  "opnieuw","opzij","over","overal","overeind","overige","overigens","paar","pas","per", \

                  "precies","recent","redelijk","reeds","rond","rondom","samen","sedert","sinds","sindsdien",\

                  "slechts","sommige","spoedig","steeds","tamelijk","te","tegen","tegenover","tenzij", \

                  "terwijl","thans","tien","tiende","tijdens","tja","toch","toe","toen","toenmaals", \

                  "toenmalig","tot","totdat","tussen","twee","tweede","u","uit","uitgezonderd","uw", \

                  "vaak","vaakwat","van","vanaf","vandaan","vanuit","vanwege","veel","veeleer","veertig", \

                  "verder","verscheidene","verschillende","vervolgens","via","vier","vierde","vijf","vijfde", \

                  "vijftig","vol","volgend","volgens","voor","vooraf","vooral","vooralsnog","voorbij", \

                  "voordat","voordezen","voordien","voorheen","voorop","voorts","vooruit","vrij", \

                  "vroeg","waar","waarom","waarschijnlijk","wanneer","want","waren","was","wat","we", \

                  "wederom","weer","weg","wegens","weinig","wel","weldra","welk","welke","werd", \

                  "werden","werder","wezen","whatever","wie","wiens","wier","wij","wijzelf","wil", \

                  "wilden","willen","word","worden","wordt","zal","ze","zei","zeker","zelf","zelfde", \

                  "zelfs","zes","zeven","zich","zichzelf","zij","zijn","zijne","zijzelf","zo","zoals", \

                  "zodat","zodra","zonder","zou","zouden","zowat","zulk","zulke","zullen","zult"]

# remove duplicates

stopword_list = list(set(stopword_list))
def lemmatize_stemming(text):

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):

    result = []

    token_list=[]

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stopword_list:

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
def remove_extrame(topic_list):

    occurence_list=[i[1] for j in topic_list for i in j]

    lambda_=np.mean(occurence_list)

    std=np.std(occurence_list)

    print(lambda_,std)

    return [i[0] for j in topic_list for i in j if i[1]>lambda_+3*std]
from numpy.random import multinomial

from numpy import log, exp

from numpy import argmax

import json



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

        for label in range(K):

            lN1 = log(m_z[label] + alpha)

            lN2 = 0

            lD2 = 0

            for word in doc:

                lN2 += log(n_z_w[label].get(word, 0) + beta)

            for j in range(1, doc_size +1):

                lD2 += log(n_z[label] + V * beta + j - 1)

            p[label] = exp(lN1 - lD1 + lN2 - lD2)



        # normalize the probability vector

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

NL_fb_df=pd.read_csv("../data/facebook_Netherlands.csv",delimiter="|",index_col=0)

NL_tw_df=pd.read_csv("../data/twitter_Netherlands.csv",delimiter="|",index_col=0)
processed_ser = NL_fb_df['message'].fillna("").apply(lambda x: re.sub(r"http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+"," ",x)).map(preprocess)
processed_docs=[item[0] for item in processed_ser]

mapping_list=[item[1] for item in processed_ser]
model=MovieGroupProcess(K=100, alpha=0.1, beta=0.01, n_iters=100)

processed_docs_short=[i[:140] for i in processed_docs]

y = model.fit(processed_docs_short,len(set(reduce(lambda x,y:x+y,processed_docs_short))))

score_list=[model.choose_best_label(i) for i in processed_docs_short]

score_with_zero=pd.DataFrame(score_list)[1].mean()

score_without_zero=pd.DataFrame([i for i in score_list if i[1]>0])[1].mean()

diff_size=len(score_list)-len([i for i in score_list if i[1]>0])

number_of_cluster=model.cluster_count

topic_list=list(reversed([sorted(x.items(), key=operator.itemgetter(1))[-5:] for x in model.cluster_word_distribution]))

#remove_list=remove_extrame(topic_list)

#print(score_with_zero,score_without_zero,diff_size,number_of_cluster,remove_list)
ind=pd.DataFrame([model.choose_best_label(i) for i in processed_docs]).sort_values([0,1],ascending=False).groupby([0]).head(1).index

for topic,string in zip(topic_list,NL_fb_df['message'].reset_index(drop=True)[ind].values):

    print(topic)

    print(string)

    print("---------------")
result_list_few=[]

count=0

for K in [100]:

    for alpha in [0.001]:

        for beta in [0.001,]:

            tmp_docs=processed_docs

            while(1):

                model=MovieGroupProcess(K=K, alpha=alpha, beta=beta, n_iters=40)

                processed_docs_short=[i[:140] for i in tmp_docs]

                y = model.fit(processed_docs_short,len(set(reduce(lambda x,y:x+y,processed_docs_short))))

                score_list=[model.choose_best_label(i) for i in processed_docs_short]

                score_with_zero=pd.DataFrame(score_list)[1].mean()

                score_without_zero=pd.DataFrame([i for i in score_list if i[1]>0])[1].mean()

                diff_size=len(score_list)-len([i for i in score_list if i[1]>0])

                number_of_cluster=model.cluster_count

                topic_list=list(reversed([sorted(x.items(), key=operator.itemgetter(1))[-5:] for x in model.cluster_word_distribution]))

                remove_list=remove_extrame(topic_list)

                print(score_with_zero,score_without_zero,diff_size,number_of_cluster,remove_list)

                if count==1:

                    break

                tmp_docs=[[word for word in doc if not word in remove_list] for doc in tmp_docs]

                count+=1

                

            print((K,alpha,beta,score_with_zero,score_without_zero,diff_size,number_of_cluster))

            result_list_few.append((K,alpha,beta,score_with_zero,score_without_zero,diff_size,number_of_cluster))
import pickle
pickle.dumps(model)
with open('NL_model.pickle', 'wb') as f:

    # Pickle the 'data' dictionary using the highest protocol available.

    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)