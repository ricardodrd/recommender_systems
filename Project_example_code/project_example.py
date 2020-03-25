#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 13:48:20 2019

@author: zhanglemei and peng
"""

import json
import os
import pandas as pd
import numpy as np
import ExplicitMF as mf
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import GridSearchCV
from surprise import KNNWithMeans


from nltk.corpus import stopwords 
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def traverse_dir(rootDir, level=2):
    dir_list = []
    print (">>>",rootDir)
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        if level == 1:
            dir_list.append(path)
        else:
            if os.path.isdir(path):
                temp_list = traverse_dir(path, level)
                dir_list.extend(temp_list)
            else:
                dir_list.append(path)
    return dir_list

def load_data(rootpath, flist):
    """
        Load events from files and convert to dataframe.
    """
    map_lst = []
    for f in flist:
        fname = os.path.join(rootpath, f)
        for line in open(fname):
            obj = json.loads(line.strip())
            if not obj is None:
                map_lst.append(obj)
    return pd.DataFrame(map_lst)

def statistics(df):
    """
        Basic statistics based on loaded dataframe
    """
    total_num = df.shape[0]
    
    print("Total number of events(front page incl.): {}".format(total_num))
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
    df_ref = df[df['documentId'].notnull()]
    num_act = df_ref.shape[0]
    
    print("Total number of events(without front page): {}".format(num_act))
    num_docs = df_ref['documentId'].nunique()
    
    print("Total number of documents: {}".format(num_docs))
    print('Sparsity: {:4.3f}%'.format(float(num_act) / float(1000*num_docs) * 100))
    df_ref.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    print("Total number of events(drop duplicates): {}".format(df_ref.shape[0]))
    print('Sparsity (drop duplicates): {:4.3f}%'.format(float(df_ref.shape[0]) / float(1000*num_docs) * 100))
    print("Describe active time:")
    print(df_ref['activeTime'].describe())
    df_active1 = df_ref[df_ref['activeTime'].notnull()]
    print(df_active1.shape) 
    df_active = df_ref[(df_ref['activeTime'].notnull()) & (df_ref['activeTime']<250)]
    print(df_active.shape)
    
    # sns.distplot(df_active['activeTime'])
    plt.hist(df_active1['activeTime'], bins=20, color='lightseagreen')
    plt.show()
    plt.boxplot(df_active1['activeTime'])
    plt.show()
    user_df = df_ref.groupby(['userId']).size().reset_index(name='counts')
    print(user_df.head())
    print("Describe by user:")
    print(user_df.describe())
    exit()
        
def load_time(df):
    #  
    df_activetime = df[(df['documentId'].notnull())& (df['activeTime'].notnull())]
    df_activetime.drop_duplicates(subset=['userId', 'documentId','activeTime'], inplace=True)
    df_activetime = df_activetime[['userId','documentId','activeTime']]
    df_activetime = df_activetime.groupby(['userId','documentId'], sort=False)['activeTime'].max().reset_index()
    df_activetime.loc[df_activetime['activeTime'] <= 10, 'rating'] = 1
    df_activetime.loc[df_activetime['activeTime'] > 10, 'rating'] = 2
    reader = Reader(rating_scale=(1, 2))
    data = Dataset.load_from_df(df_activetime[["userId", "documentId", "rating"]], reader)
    sim_options = {
        "name": ["msd", "cosine"],
        "min_support": [3, 4, 5],
        "user_based": [False, True],
    }
    param_grid = {"sim_options": sim_options}

    gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3)
    gs.fit(data)
    print(gs.best_score["rmse"])
    print(gs.best_params["rmse"])

    exit()

def load_dataset(df):
    """
        Convert dataframe to user-item-interaction matrix, which is used for 
        Matrix Factorization based recommendation.
        In rating matrix, clicked events are refered as 1 and others are refered as 0.
    """
    df = df[~df['documentId'].isnull()]
    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    df = df.sort_values(by=['userId', 'time'])
    n_users = df['userId'].nunique()
    n_items = df['documentId'].nunique()

    ratings = np.zeros((n_users, n_items))
    
    new_user = df['userId'].values[1:] != df['userId'].values[:-1]
    # print(df['userId'].values[1:])
    # print(df['userId'].values[:-1])
    print(np.unique(new_user, return_counts=True))
    print(df.columns.values)
    new_user = np.r_[True, new_user]
    print(np.unique(new_user, return_counts=True))
    df['uid'] = np.cumsum(new_user)
    # print(df['uid'].nunique())
    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame({'documentId':item_ids, 'tid':range(1,len(item_ids)+1)})
    
    # print(df.head())
    df = pd.merge(df, new_df, on='documentId', how='outer')
    df_ext = df[['uid', 'tid']]
    print(df_ext)
    
    for row in df_ext.itertuples():
        ratings[row[1]-1, row[2]-1] = 1.0
    return ratings
    
def train_test_split(ratings, fraction=0.2):
    """Leave out a fraction of dataset for test use"""
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        size = int(len(ratings[user, :].nonzero()[0]) * fraction)
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=size, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
    return train, test

def evaluate(pred, actual, k):
    """
    Evaluate recommendations according to recall@k and ARHR@k
    """
    total_num = len(actual)
    tp = 0.
    arhr = 0.
    for p, t in zip(pred, actual):
        if t in p:
            tp += 1.
            arhr += 1./float(p.index(t) + 1.)
    recall = tp / float(total_num)
    print("Recall@{} is {:.4f}".format(k, recall))
    print("ARHR@{} is {:.4f}".format(k, arhr))
    

def content_processing(df):
    """
        Remove events which are front page events, and calculate cosine similarities between
        items. Here cosine similarity are only based on item category information, others such
        as title and text can also be used.
        Feature selection part is based on TF-IDF process.
    """
    df = df[df['documentId'].notnull()]
    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    df['category'] = df['category'].str.split('|')
    df['category'] = df['category'].fillna("").astype('str')
    df['title'] = df['title'].str.split(' ')
    df['title'] = df['title'].fillna("").astype('str')
    # print(df[['title']].head())
    
    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame({'documentId':item_ids, 'tid':range(1,len(item_ids)+1)})
    df = pd.merge(df, new_df, on='documentId', how='outer')
    
    df_item = df[['tid', 'category']].drop_duplicates(inplace=False)
    df_title = df[['tid', 'title']].drop_duplicates(inplace=False)
    df_item.sort_values(by=['tid', 'category'], ascending=True, inplace=True)
    df_title.sort_values(by=['tid', 'title'], ascending=True, inplace=True)
    # select features/words using TF-IDF 
    # ngram_range(1,2): Consider unigrams and brigras    
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0)
    # Import a list of common stopwords in norwegian
    stopWordsNorsk = set(stopwords.words('norwegian'))
    tf_title = TfidfVectorizer(stop_words= stopWordsNorsk, analyzer='word',ngram_range=(1, 2),min_df=0)
    tfidf_matrix = tf.fit_transform(df_item['category'])
    tfidf = tf.fit(df_item['category'])
    # print(tfidf.vocabulary_)
    print('Dimension of feature vector: {}'.format(tfidf_matrix.shape))
    # measure similarity of two articles with cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    print("Similarity Matrix:")
    
    return cosine_sim, df

def content_recommendation(df, k=20):
    """
        Generate top-k list according to cosine similarity
    """
    cosine_sim, df = content_processing(df)
    df = df[['userId','time', 'tid', 'title', 'category']]
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
    print(df[:20]) # see how the dataset looks like
    pred, actual = [], []
    puid, ptid1, ptid2 = None, None, None
    for row in df.itertuples():
        uid, tid = row[1], row[3]
        if uid != puid and puid != None:
            idx = ptid1
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:k+1]
            sim_scores = [i for i,j in sim_scores]
            pred.append(sim_scores)
            actual.append(ptid2)
            puid, ptid1, ptid2 = uid, tid, tid
        else:
            ptid1 = ptid2
            ptid2 = tid
            puid = uid
    
    evaluate(pred, actual, k)
    
    
def collaborative_filtering(df):
    # get rating matrix
    ratings = load_dataset(df)

    # split ratings into train and test sets
    train, test = train_test_split(ratings, fraction=0.2)
    # print(train.shape)
    # print(test.shape)
    # print(np.array_equal(train, test))
# 
    # train and test model with matrix factorization
    mf_als = mf.ExplicitMF(train, n_factors=40, 
                           user_reg=0.0, item_reg=0.0)
    iter_array = [1, 2, 5, 10, 25, 50, 100]
    mf_als.calculate_learning_curve(iter_array, test)

    
    # plot out learning curves
    plot_learning_curve(iter_array, mf_als)
    

def plot_learning_curve(iter_array, model):
    """ Plot learning curves (hasn't been tested) """
    plt.plot(iter_array, model.train_mse, \
             label='Training', linewidth=5)
    plt.plot(iter_array, model.test_mse, \
             label='Test', linewidth=5)

    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    plt.xlabel('iterations', fontsize=30);
    plt.ylabel('MSE', fontsize=30);
    plt.legend(loc='best', fontsize=20);
    

if __name__ == '__main__':
    fpath = '../active1000'
    flist = traverse_dir(fpath)
    df = load_data(fpath, flist)
    ###### Get Statistics from dataset ############
    print("Basic statistics of the dataset...")
    #statistics(df)
    #load_time(df)
    ###### Recommendations based on Collaborative Filtering (Matrix Factorization) #######
    print("Recommendation based on MF...")
    collaborative_filtering(df)
    
    ###### Recommendations based on Content-based Method (Cosine Similarity) ############
    #print("Recommendation based on content-based method...")
    #content_recommendation(df, k=10)
    
    
    
    
    
    
    
    
    
    
    