import pandas as pd

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

import gensim

import numpy as np

np.random.seed(2018)

import nltk

nltk.download('wordnet')

stemmer = SnowballStemmer("english")
def lemmatize_stemming(text):

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):

    result = []

    token_list=[]

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 :

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

    mapping12many=mapping_pairs.groupby(by=[0,1]).count().reset_index().sort_values(by=[0,'count'],ascending=False)

    return mapping121,mapping12many
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

def remove_extrame(topic_list):

    occurence_list=[i[1] for j in topic_list for i in j]

    lambda_=np.mean(occurence_list)

    std=np.std(occurence_list)

    print(lambda_,std)

    return [i[0] for j in topic_list for i in j if i[1]>lambda_+3*std]
given_topic_df=pd.read_csv("../data/2030_goals.csv")
processed_ser=given_topic_df['description'].map(preprocess)

processed_docs=[item[0] for item in processed_ser]

mapping_list=[item[1] for item in processed_ser]



mapping121_description,mapping12many_description=produce_mapping(mapping_list)



from sklearn.feature_extraction.text import TfidfVectorizer



tf_transform=TfidfVectorizer(max_df=0.95, min_df=2,ngram_range=(1,1))



tf = tf_transform.fit_transform([" ".join(i) for i in processed_docs])



tf_transform.get_feature_names()



from numpy import argmax

description_list=[]

index=0

for i in np.array(tf.todense()):

    print(i.shape)

    print([list(j)+[given_topic_df['topic'][index]] for j in sorted(list(zip(i,tf_transform.get_feature_names())))[-5:]])

    description_list+=[list(j)+[given_topic_df['topic'][index]] for j in sorted(list(zip(i,tf_transform.get_feature_names())))[-5:]]

    index+=1
description_df=pd.DataFrame(description_list,columns=['score',"stem",'topic'])

description_df['type']=1
processed_ser=given_topic_df['strategy'].map(preprocess)

processed_docs=[item[0] for item in processed_ser]

mapping_list=[item[1] for item in processed_ser]



mapping121_strategy,mapping12many_strategy=produce_mapping(mapping_list)



from sklearn.feature_extraction.text import TfidfVectorizer



tf_transform=TfidfVectorizer(max_df=0.95, min_df=2,ngram_range=(1,1))



tf = tf_transform.fit_transform([" ".join(i) for i in processed_docs])



tf_transform.get_feature_names()



from numpy import argmax

strategy_list=[]

index=0

for i in np.array(tf.todense()):

    print(i.shape)

    print([list(j)+[given_topic_df['topic'][index]] for j in sorted(list(zip(i,tf_transform.get_feature_names())))[-5:]])

    strategy_list+=[list(j)+[given_topic_df['topic'][index]] for j in sorted(list(zip(i,tf_transform.get_feature_names())))[-5:]]

    index+=1
given_topic_df['topic'][1]
strategy_df=pd.DataFrame(strategy_list,columns=['score',"stem",'topic'])

strategy_df['type']=2
processed_ser=given_topic_df['extra'].map(preprocess)

processed_docs=[item[0] for item in processed_ser]

mapping_list=[item[1] for item in processed_ser]



mapping121_extra,mapping12many_extra=produce_mapping(mapping_list)



from sklearn.feature_extraction.text import TfidfVectorizer



tf_transform=TfidfVectorizer(max_df=0.95, min_df=2,ngram_range=(1,1))



tf = tf_transform.fit_transform([" ".join(i) for i in processed_docs])



tf_transform.get_feature_names()



from numpy import argmax

extra_list=[]

index=0

for i in np.array(tf.todense()):

    print(i.shape)

    print([list(j)+[given_topic_df['topic'][index]] for j in sorted(list(zip(i,tf_transform.get_feature_names())))[-5:]])

    extra_list+=[list(j)+[given_topic_df['topic'][index]] for j in sorted(list(zip(i,tf_transform.get_feature_names())))[-5:]]

    index+=1
extra_df=pd.DataFrame(extra_list,columns=['score',"stem",'topic'])

extra_df['type']=3
given_topic=pd.concat([description_df,strategy_df,extra_df],axis=0).reset_index(drop=True)
given_topic
MI_fb_df=pd.read_csv("../data/facebook_Malawi.csv",delimiter="|",index_col=0)

MI_tw_df=pd.read_csv("../data/twitter_Malawi.csv",delimiter="|",index_col=0)
processed_ser = MI_fb_df['message'].fillna("").apply(lambda x: re.sub(r"http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+"," ",x)).map(preprocess)
processed_docs=[item[0] for item in processed_ser]

mapping_list=[item[1] for item in processed_ser]

mapping121_MI,mapping12many_MI=produce_mapping(mapping_list)
from functools import reduce

import operator

model=MovieGroupProcess(K=30, alpha=0.01, beta=0.01, n_iters=200)

processed_docs_short=[i[:140] for i in processed_docs]

y = model.fit(processed_docs_short,len(set(reduce(lambda x,y:x+y,processed_docs_short))))

score_list=[model.choose_best_label(i) for i in processed_docs_short]

topic_list=list(reversed([sorted(x.items(), key=operator.itemgetter(1))[-5:] for x in model.cluster_word_distribution if len(x)]))

remove_list=remove_extrame(topic_list)
topic_list
tmp_docs=[[word for word in doc if not word in remove_list] for doc in processed_docs]
model=MovieGroupProcess(K=30, alpha=0.5, beta=0.1, n_iters=200)

processed_docs_short=[i[:140] for i in tmp_docs]

y = model.fit(processed_docs_short,len(set(reduce(lambda x,y:x+y,processed_docs_short))))

score_list=[model.choose_best_label(i) for i in processed_docs_short]

topic_list=list(reversed([sorted(x.items(), key=operator.itemgetter(1))[-5:] for x in model.cluster_word_distribution if len(x)]))

print(topic_list)
pd.DataFrame(topic_list)
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
processed_ser=given_topic_df['description'].map(preprocess)

processed_docs=[item[0] for item in processed_ser]

mapping_list=[item[1] for item in processed_ser]
dictionary = gensim.corpora.Dictionary(processed_docs)

dictionary.filter_extremes()

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
dictionary=dictionary.from_documents(processed_docs)
bow_corpus
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
MI_topic_df=pd.DataFrame([[(mapping121_MI[mapping121_MI[0]==j[0]][1].values[0],j[1]) for j in i] for i in topic_list])
MI_topic_df
def stem2original(row):

    stem,_type=row['stem'],row['type']

    if _type==1:

        return mapping121_description[mapping121_description[0]==stem][1].values[0]

    elif _type==2:

        return mapping121_strategy[mapping121_strategy[0]==stem][1].values[0]

    elif _type==3:

        return mapping121_extra[mapping121_extra[0]==stem][1].values[0]
given_topic['original_word']=given_topic.apply(lambda x:stem2original(x),axis=1)
given_topic
from fuzzywuzzy import fuzz
weight_description=1.0

weight_strategy=0.5

weight_extra=0.3

weight_list=[weight_description,weight_strategy,weight_extra]

doc_count=list(reversed([i for i in model.cluster_doc_count if i]))

def scoring(given_topic,social_media):

    #tmp=social_media.copy()

    #print(social_media.values)

    score_list=[]

    given_topic_list=given_topic['topic'].drop_duplicates().to_list()

    for i in social_media.iterrows():

        tmp=given_topic.copy()

        tmp['score_topic']=given_topic.apply(lambda x:score_per_topic(x,i[1].tolist()),axis=1)

        

        score_list.append([i[0]]+tmp.groupby(by=['topic']).sum()["score_topic"].tolist()+[doc_count[i[0]]]+[i[0] for i in social_media.values.tolist()[i[0]]])

    return(pd.DataFrame(score_list,columns=["index"]+given_topic_list+["NoDoc"]+["key_word "+str(i) for i in range(5,0,-1)]))

#scoring(given_topic,MI_topic_df).to_csv("../data/MI_topic_matching.csv")

result=scoring(given_topic,MI_topic_df)
print(result.to_latex())
def score_per_topic(row,word_list):

    for word in word_list:

        if fuzz.ratio(word[0],row['original_word'])>90:

            return word[1]*weight_list[row['type']-1]

    return 0

given_topic.apply(lambda x:score_per_topic(x,[('people', 155), ('support', 167), ('disaster', 188), ('society', 238), ('said', 257)]),axis=1).sum()
list(reversed([i for i in model.cluster_doc_count if i]))
MI_topic_df.values.tolist()
%%javascript

require.config({

    paths: {

        d3: 'https://d3js.org/d3.v5.min'

    }

});
%%javascript

(function(element) {

    require(['d3'], function(d3) {   

        var data = [1, 2, 4, 8, 16, 8, 4, 2, 1]



        var svg = d3.select(element.get(0)).append('svg')

            .attr('width', 400)

            .attr('height', 200);

        svg.selectAll('circle')

            .data(data)

            .enter()

            .append('circle')

            .attr("cx", function(d, i) {return 40 * (i + 1);})

            .attr("cy", function(d, i) {return 100 + 30 * (i % 3 - 1);})

            .style("fill", "#1570a4")

            .transition().duration(2000)

            .attr("r", function(d) {return 2*d;})

        ;

    })

})(element);
import pandas as pd

from tabulate import tabulate



def pandas_df_to_markdown_table(df):

    # Dependent upon ipython

    # shamelessly stolen from https://stackoverflow.com/questions/33181846/programmatically-convert-pandas-dataframe-to-markdown-table

    from IPython.display import Markdown, display

    fmt = ['---' for i in range(len(df.columns))]

    df_fmt = pd.DataFrame([fmt], columns=df.columns)

    df_formatted = pd.concat([df_fmt, df])

    #display(Markdown(df_formatted.to_csv(sep="|", index=False)))

    return Markdown(df_formatted.to_csv(sep="|", index=False))

#     return df_formatted



def df_to_markdown(df, y_index=False):

    blob = tabulate(df, headers='keys', tablefmt='pipe')

    if not y_index:

        # Remove the index with some creative splicing and iteration

        return '\n'.join(['| {}'.format(row.split('|', 2)[-1]) for row in blob.split('\n')])

    return blob

print(df_to_markdown(result))