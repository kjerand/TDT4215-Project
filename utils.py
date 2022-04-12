import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cbf import content_based_filtering

def load_data(path):
    """
        Load events from files and convert to dataframe.
    """
    map_lst=[]
    for f in os.listdir(path):
        file_name=os.path.join(path,f)
        if os.path.isfile(file_name):
            for line in open(file_name):
                obj = json.loads(line.strip())
                if not obj is None:
                    map_lst.append(obj)
    return pd.DataFrame(map_lst) 

def load_dataset(df):
    """
        Convert dataframe to user-item-interaction matrix, which is used for 
        Matrix Factorization based recommendation.
        In rating matrix, clicked events are refered as 1 and others are refered as 0.
    """
    df = df[~df['documentId'].isnull()]
    df = df.drop_duplicates(subset=['userId', 'documentId']).reset_index(drop=True)
    df = df.sort_values(by=['userId', 'time'])
    n_users = df['userId'].nunique()
    n_items = df['documentId'].nunique()

    ratings = np.zeros((n_users, n_items))
    new_user = df['userId'].values[1:] != df['userId'].values[:-1]
    new_user = np.r_[True, new_user]
    df['uid'] = np.cumsum(new_user)
    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame({'documentId':item_ids, 'tid':range(1,len(item_ids)+1)})
    df = pd.merge(df, new_df, on='documentId', how='outer')
    df_ext = df[['uid', 'tid']]
    
    for row in df_ext.itertuples():
        ratings[row[1]-1, row[2]-1] = 1.0
    return ratings


def cbf_plot_no_of_feature(df, knn):
    no_of_features = [30, 50, 60, 75, 100, 200, 300, 500]
    recall_vals = []
    arhr_vals = []

    for f in no_of_features:
        print("\nFeatures: ", f,"\n")

        recall, arhr = content_based_filtering(df, 20, f, knn)
        recall_vals.append(recall)
        arhr_vals.append(arhr)

    plot_learning_curve(no_of_features, recall_vals, arhr_vals)



def plot_learning_curve(iter_array, recall, arhr):
    """ Plot learning curves """
    plt.plot(iter_array, recall, \
             label='Recall', linewidth=5)
    plt.plot(iter_array, arhr, \
             label='ARHR', linewidth=5)

    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    plt.xlabel('features', fontsize=20);
    plt.ylabel('score', fontsize=20);
    plt.legend(loc='best', fontsize=18);
    plt.show()
