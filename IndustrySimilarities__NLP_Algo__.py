# -*- coding: utf-8 -*-
"""
@abstract: 
    NLP algorithm (semantic analysis) given an SIC_Indutry_name produces a similarity vector to all other industries
    using cosine distance. Industries which do not meet a ceratin threshold of similarity are filtered in the function

@author: 
    Bratislav Petkovic

@references:
    https://stackoverflow.com/questions/65199011/is-there-a-way-to-check-similarity-between-two-full-sentences-in-python
    https://betterprogramming.pub/introduction-to-gensim-calculating-text-similarity-9e8b55de342d

"""
#______________________________________Preprocessings______________________________________
import pandas as pd
from nltk.corpus import stopwords
from nltk import download
import numpy as np
industrySet = pd.read_csv("Z:/__Bratislav_Petkovic/RenewalGenius/Industries_Study/IndustryGrouping.csv")
industrySet = pd.read_excel("Z:/__Bratislav_Petkovic/RenewalGenius/Industries_Study/IndustryGroupingFullDataset.xlsx")

industrySet = industrySet.sort_values('SIC_INDUSTRY_CLASS_NM', ascending=True)
industrySet_alpha = industrySet.set_index([pd.Index([*range(0,len(industrySet), 1) ])], 'SIC_INDUSTRY_CLASS_NM')


# removing stop words 
industrySet_alpha["sub_industry_filtered"] = [phrase.replace(',',"") for phrase in industrySet_alpha["SIC_INDUSTRY_CLASS_NM"]]
documents = list(industrySet_alpha.sub_industry_filtered)
stoplist = set(['or', ',']) #'product', 'products', 'service', 'services', 'of'])
stop_words = stopwords.words('english')
stop_words_mine = [ 'goods','nec', '&', 'services', 'store', 'equipment', 'products', 'services,', 'commercial', "agencies", 
                   "preparation","preparations","prepare","agency", "products,", "other", "goods", "national", "protection", "protecting", "bodies",
                   "plans" ,"mineral", "minerals", "misc.", "miscellaneous", "clinics", "specialty", "special", "mills", "specialties", "exc.", "ex."
                   "supply","supplies", "stores" ,"system" ,"systems", "allied", "public", "primary", "facilities", "centers", "related", "shops"]
stop_words_new = list(set(stop_words + stop_words_mine))
texts = [  [word.lower() for word in document.split() if word.lower() not in stop_words_new]  for document in documents]

industrySet_alpha["sub_industry_filtered"] = [' '.join(word_arr) for word_arr in  texts]


#______________________________________TENSOR FLOW APPROACH______________________________________

import tensorflow_hub as hub
from scipy.spatial import distance
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")  # delet folder in file explorer if (OS) error 

def get_similar_industries_ten_flow(given_industry, industry_df, similarity_threshold= 0.0, pool_threshold = 10):
    
    data = {'ComparerSubIndustry': [],'Similarity': []}
    processed_sub_ind = industry_df[industry_df["SIC_INDUSTRY_CLASS_NM"] == given_industry]["sub_industry_filtered"].iloc[0]
    # print(processed_sub_ind)
    for index, row in industry_df.iterrows():
        # print(row.SIC_INDUSTRY_CLASS_NM)
        # print(row.Industry)
        embeddings = embed([processed_sub_ind, row.sub_industry_filtered])
        data["ComparerSubIndustry"].append(row.SIC_INDUSTRY_CLASS_NM)
        # data["ComparerIndustry"].append(row.Industry)
        data["Similarity"].append(1 - distance.cosine(embeddings[0], embeddings[1]))
 
    resultDF = pd.DataFrame(data)
    resultDF = resultDF[resultDF["Similarity"] > similarity_threshold]
    resultDF = resultDF.sort_values('Similarity', ascending=False)
    resultDF = resultDF.head(pool_threshold)
    resultDF.ComparerSubIndustry = pd.Series(resultDF.ComparerSubIndustry, dtype="string") 
    return resultDF
    
    
    
beer_and_ale = get_similar_industries_ten_flow("Beer and ale", industrySet_alpha)
advertising_agencies_2 = get_similar_industries_ten_flow("Advertising agencies", industrySet_alpha, pool_threshold = 25)





#___________________________________GenSim Approach 3_______________________________________________
#https://radimrehurek.com/gensim/auto_examples/tutorials/run_scm.html
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim.downloader as api #api.info()
model = api.load('word2vec-google-news-300') 
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

# filter any stop words 
# download('stopwords')  # Download stopwords list.

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words_new]

phrase_vector = [preprocess(sub_industry) for sub_industry in list(industrySet_alpha.sub_industry_filtered)]

# create a Dictionary and fill it to 
dictionary = Dictionary(phrase_vector)
documents = [ dictionary.doc2bow(sub_industry) for sub_industry in phrase_vector]

#create a model from the dictionary of documents and apply to everything
tfidf = TfidfModel(documents)
tfidf_phrases = [  tfidf[sub_industry] for sub_industry in documents]

# Create a similarity matrix using the model-embedded words
termsim_index = WordEmbeddingSimilarityIndex(model)
termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)

def get_similar_industries_gen_sim(given_industry, industry_df):
    sic_industry_index = industry_df.index[industry_df['SIC_INDUSTRY_CLASS_NM'] == given_industry].tolist()[0]
    data = {'ComparerSubIndustry': [],'Similarity': []}
    for i in list(industry_df.SIC_INDUSTRY_CLASS_NM):
        comparer_industry_index = industry_df.index[industry_df['SIC_INDUSTRY_CLASS_NM'] == i].tolist()[0]
        similarity = termsim_matrix.inner_product(tfidf_phrases[sic_industry_index], tfidf_phrases[comparer_industry_index], normalized=(True, True))
        # print('similarity = %.4f' % similarity)
        data['ComparerSubIndustry'].append(i)
        data['Similarity'].append(similarity)
        # data["ComparerIndustry"].append(industry_df[industry_df["SIC_INDUSTRY_CLASS_NM"] == given_industry]["Industry"].iloc[0])
        
    gen_sim_result_df = pd.DataFrame(data)
    gen_sim_result_df = gen_sim_result_df[gen_sim_result_df.Similarity > 0]
    gen_sim_result_df = gen_sim_result_df.sort_values('Similarity', ascending=False)
    gen_sim_result_df.ComparerSubIndustry = pd.Series(gen_sim_result_df.ComparerSubIndustry, dtype="string") 

    return gen_sim_result_df



advertising_agencies = get_similar_industries_gen_sim("Advertising agencies", industrySet_alpha)
advertising = get_similar_industries_gen_sim("Advertising, nec", industrySet_alpha)
beer_and_ale = get_similar_industries_gen_sim("Beer and ale", industrySet_alpha)





# time complexity == O(n*(2*n) == 2n^2), 
# returns a dictionary of dataframes where keys are industries and values are dataframe results
def get_all_similar_industries(industry_df, algo_weights=[0.5,0.5]):
    full_df = pd.DataFrame()
    df_dict = {}
    counter = 0
    for sub_industry in list(industry_df.SIC_INDUSTRY_CLASS_NM): 
        print(sub_industry)
        # run gen sim algo, filter out all zeros 
        gen_sim_res = get_similar_industries_gen_sim(sub_industry, industry_df)
        
        # run tensor_flow algo, get first 5-10?
        ten_flow_res = get_similar_industries_ten_flow(sub_industry, industry_df)
        
        # join_results = gen_sim_res.merge(ten_flow_res, on = ['ComparerSubIndustry'])                                                        #INNER JOIN
        join_results = gen_sim_res.merge(ten_flow_res, suffixes=("__GS__", "__TF__"), how='outer', on = ['ComparerSubIndustry'])              #OUTER JOIN

        # if they dont have any matching ones, provide two sets of results ? 
        if(len(join_results) < 2):
            print(" I am here uh oh")
            # do a left merge on each and see which one is better ? 
        join_results = join_results.replace(np.nan, 0)
        join_results["SIC_Industry"] = sub_industry
        # join_results["ComparerIndustry"] = np.where(join_results.ComparerIndustry__GS__.notnull(),join_results.ComparerIndustry__GS__, join_results.ComparerIndustry__TF__ )
        join_results["Combo_Similarity"] = join_results['Similarity__GS__'] * algo_weights[0] + join_results['Similarity__TF__'] * algo_weights[1]
        
        join_results = join_results[['SIC_Industry','ComparerSubIndustry', 'Similarity__GS__', 'Similarity__TF__', 'Combo_Similarity']]
        join_results = join_results.sort_values('Combo_Similarity', ascending=False)

        df_dict[sub_industry] = join_results
        full_df = pd.concat([full_df, join_results], ignore_index = True)
        counter+=1
        # if(counter>10):
        #     break
    full_df.to_csv("Z:/__Bratislav_Petkovic/RenewalGenius/Industries_Study/NLP_Results.csv",index=False)
    return df_dict


final_similarity_res = get_all_similar_industries(industrySet_alpha)









# #___________________________________GenSim Approach 2_______________________________________________

from gensim import corpora, models, similarities
from collections import defaultdict

documents = list(industrySet.SIC_INDUSTRY_CLASS_NM)

stoplist = set(['or', ',']) #'product', 'products', 'service', 'services', 'of'])

texts = [[word.lower() for word in document.split()
          if word.lower() not in stoplist]
          for document in documents]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 0]
          for text in texts]

dictionary = corpora.Dictionary(texts)


corpus = [dictionary.doc2bow(text) for text in texts]
lsi = models.LsiModel(corpus, id2word=dictionary)
doc = "Beer and ale"
vec_bow = dictionary.doc2bow(doc.lower().split())

# convert the query to LSI space
vec_lsi = lsi[vec_bow]
index = similarities.MatrixSimilarity(lsi[corpus])

# perform a similarity query against the corpus
sims = index[vec_lsi]
sims = sorted(enumerate(sims), key=lambda item: -item[1])


index_list = [*range(0,len(industrySet), 1) ]
indsutrySet2 = industrySet.set_index([pd.Index(index_list)], 'SIC_INDUSTRY_CLASS_NM')
data = {'ComparerSubIndustry': [],'Similarity': []}
for i in range(0,14):
    data['ComparerSubIndustry'].append(industrySet.iloc[int(sims[i][0])].SIC_INDUSTRY_CLASS_NM)
    data['Similarity'].append(sims[i][1])
    

resultDF = pd.DataFrame(data)

# # LSI is not good enough bc it derives semantics from the word count in the corpus. We would need a much 
# # bigger corpus and it doesn't look at the doc hollistically  








# #___________________________________GenSim Approach_______________________________________________
from gensim import corpora, models, similarities
import jieba
texts = ['Beer and ale ']

keyword = 'Japan has some great novelists. Who is your favorite Japanese writer?'
texts = [jieba.lcut(text) for text in texts]
dictionary = corpora.Dictionary(texts)
feature_cnt = len(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus) 
kw_vector = dictionary.doc2bow(jieba.lcut(keyword))
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features = feature_cnt)
sim = index[tfidf[kw_vector]]
for i in range(len(sim)):
    print('keyword is similar to text%d: %.2f' % (i + 1, sim[i]))

# Not IT DUDE> THATS A CORPUS( A COMPLETELY DIFFERENT BEAST)
















