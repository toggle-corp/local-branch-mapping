from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation

import pandas as pd

pd.set_option('display.expand_frame_repr', False)

import re

import math
NL_fb_df=pd.read_csv("../data/facebook_Netherlands.csv",delimiter="|",index_col=0)

NL_tw_df=pd.read_csv("../data/twitter_Netherlands.csv",delimiter="|",index_col=0)
NL_fb_df.columns
NL_tw_df.columns
NL_tw_df.fillna("")
tf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,ngram_range=(1,3))

tf = tf_vectorizer.fit_transform(NL_tw_df['text'].fillna("").apply(lambda x: re.sub(r"http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+"," ",x)))
lda = LatentDirichletAllocation(n_components=10, max_iter=5,

                                learning_method='online',

                                learning_offset=50.,

                                random_state=0)
lda.fit(tf)
def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        message = "Topic #%d: " % topic_idx

        message += ",".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        print(message)

    print()

n_top_words=5
print_top_words(lda, tf_vectorizer.get_feature_names(), n_top_words)
lda.components_.shape
lda.components_
MI_fb_df=pd.read_csv("../data/facebook_Malawi.csv",delimiter="|",index_col=0)

MI_tw_df=pd.read_csv("../data/twitter_Malawi.csv",delimiter="|",index_col=0)
pd.read_csv("../data/twitter_Malawi.csv",delimiter="|",index_col=0).shape
tf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,ngram_range=(1,3),stop_words = 'english')

tf = tf_vectorizer.fit_transform(MI_fb_df['message'].fillna("").apply(lambda x: re.sub(r"http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+"," ",x)))
MI_fb_df.columns
lda = LatentDirichletAllocation(n_components=5, max_iter=5,

                                learning_method='online',

                                learning_offset=50.,

                                random_state=0)
lda.fit(tf)
print_top_words(lda, tf_vectorizer.get_feature_names(), n_top_words)
nmf = NMF(n_components=5, random_state=1,

          alpha=.1, l1_ratio=.5).fit(tf)
print_top_words(nmf, tf_vectorizer.get_feature_names(), n_top_words)
from gensim.models.coherencemodel import CoherenceModel
import gensim

import numpy as np

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

np.random.seed(2018)

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
processed_ser = MI_fb_df['message'].fillna("").apply(lambda x: re.sub(r"http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+"," ",x)).map(preprocess)
processed_docs=[item[0] for item in processed_ser]

mapping_list=[item[1] for item in processed_ser]
mapping121,mapping12many=produce_mapping(mapping_list)
dictionary = gensim.corpora.Dictionary(processed_docs)

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:

    pprint(doc)

    break
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary, passes=2, workers=2)
cm = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')

print( cm.get_coherence())
for k in [3,5,10,15]:

    for decay in [0.5, 0.7, 0.9]:

        print(k,decay)

        coherence_list=[]

        for i in range(5):

            lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=k, id2word=dictionary, passes=2, workers=5,decay=decay,eval_every=10 ,random_state=i)

            cm = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')

            coherence_list.append(cm.get_coherence())

        print(coherence_list)

        print(np.std(coherence_list))

        print(sum(coherence_list)/len(coherence_list))
for k in [3,5,10,15]:

    for decay in [0.5, 0.7, 0.9]:

        for iterations in [50,100,200]:

            print(k,decay,iterations )

            coherence_list=[]

            for i in range(5):

                lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=k, id2word=dictionary, passes=2, workers=5,decay=decay,eval_every=10 ,random_state=i,iterations =iterations )

                cm = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')

                coherence_list.append(cm.get_coherence())

            print(coherence_list)

            print(np.std(coherence_list))

            print(sum(coherence_list)/len(coherence_list))
for idx, topic in lda_model.print_topics(-1):

    print('Topic: {} \nWords: {}'.format(idx, topic))
topics_assigned=[list(sorted(lda_model[corpus], key=lambda tup: -1*tup[1]))[0] for corpus in  bow_corpus]
def convert(stem,mapping_df):

    return mapping_df[mapping_df[0]==stem][1].values[0]
def parse_string(string):

    terms=string.split('+')

    terms=[t.split('*') for t in terms]

    terms=[(float(i[0]),convert(eval(i[1]),mapping121)) for i in terms]

    return terms
index=pd.DataFrame(topics_assigned).sort_values(by=[0,1],ascending=False).groupby(by=0).head(1).index
score_df=pd.DataFrame(topics_assigned)
score_df['message']=MI_fb_df['message'].reset_index(drop=True)
score_df['score']=score_df[1]/np.log(score_df['message'].fillna('').apply(lambda x:len(x) if len(x)>200 else 200 ),order=2)
index=score_df.sort_values(by=[0,'score'],ascending=False).groupby(by=0).head(1).index
score_df.sort_values(by=[0,'score'],ascending=False).groupby(by=0).head(1)
MI_fb_df['message'].reset_index(drop=True)[index].values
for topic,string in zip([(i[0],parse_string(i[1])) for i in lda_model.show_topics()],MI_fb_df['message'].reset_index(drop=True)[index].values):

    print(topic)

    print(string)

    print("---------------")
MI_fb_df.shape
MI_tw_df.shape
def parse_string_without_covert(string):

    terms=string.split('+')

    terms=[t.split('*') for t in terms]

    terms=[(float(i[0]),eval(i[1])) for i in terms]

    return terms
topic_list_stem=[(i[0],parse_string_without_covert(i[1])) for i in lda_model.show_topics()]
topic_list=[(i[0],parse_string(i[1])) for i in lda_model.show_topics()]
def calculate_score_one_topic(string,keywords):

    if len(string)==0:

        return 0

    score=0

    for word in keywords:

        weight=word[0]

        score+=weight*string.count(word[1])

        #print(score)

    if len(string)<200:

        scale=1

    else:

        scale=math.exp(len(string)/200)

    

    return score/scale
def scoring(string,topics):

    result=[]

    for num,keywords in topics:

        result.append((num,calculate_score_one_topic(string,keywords)))

    return sorted(result,key=lambda x:-1*x[1])[0]

scoring_df=pd.DataFrame(list(MI_fb_df['message'].fillna('').apply(lambda x:scoring(x,topic_list_stem)).values)).sort_values(by=[0,1],ascending=False).groupby(by=0).head(1).sort_values(by=[0],ascending=True)

scoring_df
#topic_list.reverse()

for topic,string in zip(topic_list,MI_fb_df['message'].reset_index(drop=True)[scoring_df.index].values):

    print(topic)

    print(string)

    print("---------------")
coherence_model_lda = CoherenceModel(model=model, texts=processed_docs, dictionary=dictionary, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
import mgp
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

len(bow_corpus)
len(processed_docs)
model=MovieGroupProcess(K=10, alpha=0.1, beta=0.01, n_iters=200)
y = model.fit(processed_docs,2772)
# for i in model.cluster_word_distribution:

#     print(len(i))

model.cluster_word_distribution[-1]
ind=pd.DataFrame([model.choose_best_label(i) for i in processed_docs]).sort_values([0,1],ascending=False).groupby([0]).head(1).index
import operator

topic_list=list(reversed([sorted(x.items(), key=operator.itemgetter(1))[-5:] for x in model.cluster_word_distribution]))
for i in model.cluster_word_distribution:

    print(len(i))
MI_fb_df['message'].reset_index(drop=True)[ind].values
for topic,string in zip(topic_list,MI_fb_df['message'].reset_index(drop=True)[ind].values):

    print(topic)

    print(string)

    print("---------------")
result_list=[]

for K in [5,8,10,20]:

    for alpha in [0.001,0.01,0.05,0.1,0.3,0.5]:

        for beta in [0.001,0.01,0.05,0.1,0.3,0.5]:

            model=MovieGroupProcess(K=K, alpha=alpha, beta=beta, n_iters=200)

            y = model.fit(processed_docs,2772)

            score_list=[model.choose_best_label(i) for i in processed_docs]

            score_with_zero=pd.DataFrame(score_list)[1].mean()

            score_without_zero=pd.DataFrame([i for i in score_list if i[1]>0])[1].mean()

            diff_size=len(score_list)-len([i for i in score_list if i[1]>0])

            number_of_cluster=model.cluster_count

            print(K,alpha,beta,score_with_zero,score_without_zero,diff_size,number_of_cluster)

            result_list.append((K,alpha,beta,score_with_zero,score_without_zero,diff_size,number_of_cluster))
from functools import reduce

size=len(set(reduce(lambda x,y:x+y,processed_docs)))
size
result_list_few=[]

for K in [5,8,10,20]:

    for alpha in [0.001,0.01,0.05,0.1,0.3,0.5]:

        for beta in [0.001,0.01,0.05,0.1,0.3,0.5]:

            model=MovieGroupProcess(K=K, alpha=alpha, beta=beta, n_iters=200)

            processed_docs_short=[i[:140] for i in processed_docs]

            y = model.fit(processed_docs_short,len(set(reduce(lambda x,y:x+y,processed_docs_short))))

            score_list=[model.choose_best_label(i) for i in processed_docs_short]

            score_with_zero=pd.DataFrame(score_list)[1].mean()

            score_without_zero=pd.DataFrame([i for i in score_list if i[1]>0])[1].mean()

            diff_size=len(score_list)-len([i for i in score_list if i[1]>0])

            number_of_cluster=model.cluster_count

            print(K,alpha,beta,score_with_zero,score_without_zero,diff_size,number_of_cluster)

            result_list_few.append((K,alpha,beta,score_with_zero,score_without_zero,diff_size,number_of_cluster))
len(result_list)
result_list_few=result_list[-144:]
result_list=result_list[:144]
len(result_list)
result_list_few
result_df=pd.DataFrame(result_list,columns =['k','alpha','beta','score_all','score_nonzero',"Num_of_zero",'final_k'])
result_few_df=pd.DataFrame(result_list_few,columns =['k','alpha','beta','score_all','score_nonzero',"Num_of_zero",'final_k'])
result_df.to_csv("result_gsdmm.csv")

result_few_df.to_csv("result_few_gsdmm.csv")
model=MovieGroupProcess(K=10, alpha=0.01, beta=0.01, n_iters=50)

processed_docs_short=[i[:140] for i in processed_docs]

y = model.fit(processed_docs_short,len(set(reduce(lambda x,y:x+y,processed_docs_short))))

score_list=[model.choose_best_label(i) for i in processed_docs_short]

score_with_zero=pd.DataFrame(score_list)[1].mean()

score_without_zero=pd.DataFrame([i for i in score_list if i[1]>0])[1].mean()

diff_size=len(score_list)-len([i for i in score_list if i[1]>0])

number_of_cluster=model.cluster_count

#print(K,alpha,beta,score_with_zero,score_without_zero,diff_size,number_of_cluster,model.cluster_word_distribution)

#result_list_few.append((K,alpha,beta,score_with_zero,score_without_zero,diff_size,number_of_cluster))
import operator

topic_list=list(reversed([sorted(x.items(), key=operator.itemgetter(1))[-5:] for x in model.cluster_word_distribution]))
def remove_extrame(topic_list):

    occurence_list=[i[1] for j in topic_list for i in j]

    lambda_=np.mean(occurence_list)

    std=np.std(occurence_list)

    print(lambda_,std)

    return [i[0] for j in topic_list for i in j if i[1]>lambda_+3*std]
remove_extrame(topic_list)
tmp_docs=processed_docs

while(1):

    model=MovieGroupProcess(K=10, alpha=0.01, beta=0.01, n_iters=50)

    processed_docs_short=[i[:140] for i in tmp_docs]

    y = model.fit(processed_docs_short,len(set(reduce(lambda x,y:x+y,tmp_docs))))

    score_list=[model.choose_best_label(i) for i in processed_docs_short]

    score_with_zero=pd.DataFrame(score_list)[1].mean()

    score_without_zero=pd.DataFrame([i for i in score_list if i[1]>0])[1].mean()

    diff_size=len(score_list)-len([i for i in score_list if i[1]>0])

    number_of_cluster=model.cluster_count

    topic_list=list(reversed([sorted(x.items(), key=operator.itemgetter(1))[-5:] for x in model.cluster_word_distribution]))

    remove_list=remove_extrame(topic_list)

    if len(remove_list)==0:

        break

    tmp_docs=[[word for word in doc if not word in remove_list] for doc in tmp_docs]

    print(score_with_zero,score_without_zero,diff_size,number_of_cluster,remove_list)

#result_list_few.append((K,alpha,beta,score_with_zero,score_without_zero,diff_size,number_of_cluster))
ind=pd.DataFrame([model.choose_best_label(i) for i in processed_docs]).sort_values([0,1],ascending=False).groupby([0]).head(1).index
for topic,string in zip(topic_list,MI_fb_df['message'].reset_index(drop=True)[ind].values):

    print(topic)

    print(string)

    print("---------------")
topic_list
result_list_few=[]

for K in [5,8,10,20]:

    for alpha in [0.001,0.01,0.05,0.1,0.3,0.5]:

        for beta in [0.001,0.01,0.05,0.1,0.3,0.5]:

            tmp_docs=processed_docs

            while(1):

                model=MovieGroupProcess(K=K, alpha=alpha, beta=beta, n_iters=100)

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

                if len(remove_list)==0:

                    break

                tmp_docs=[[word for word in doc if not word in remove_list] for doc in tmp_docs]

                

            print((K,alpha,beta,score_with_zero,score_without_zero,diff_size,number_of_cluster))

            result_list_few.append((K,alpha,beta,score_with_zero,score_without_zero,diff_size,number_of_cluster))
result_list_few
result_few_df=pd.DataFrame(result_list_few,columns =['k','alpha','beta','score_all','score_nonzero',"Num_of_zero",'final_k'])
result_few_df.to_csv("result_few_gsdmm_rm_extrme.csv")