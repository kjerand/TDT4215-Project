import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download("punkt")
nltk.download("stopwords")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from evaluate import evaluate

def content_processing(df, features):
    """
        Remove events which are front page events, and calculate cosine similarities between
        items. Here cosine similarity are only based on item category information, others such
        as title and text can also be used.
        Feature selection part is based on TF-IDF process.
    """
    

    df = df[df['documentId'].notnull()]
    df = df.drop_duplicates(subset=['documentId'], keep="first", inplace=False)

    stemmer = SnowballStemmer('norwegian')
    stop = stopwords.words('norwegian')
    df['title'] = df['title'].str.replace('[^\w\s]','')
    df['title'] = df['title'].str.lower()
    df['title'] = df['title'].apply(lambda x: [item for item in x.split() if item not in stop])
    df['title'] = df['title'].apply(lambda x: [stemmer.stem(y) for y in x])
    df['title'] = df['title'].fillna("").astype('str')
    
    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame({'documentId':item_ids, 'tid':range(0,len(item_ids))})
    df = pd.merge(df, new_df, on='documentId', how='outer')
    df_item = df[['tid', 'title']].drop_duplicates(inplace=False)
    df_item.sort_values(by=['tid', 'title'], ascending=True, inplace=True)

    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0,max_features=features)
    tfidf_matrix = tf.fit_transform(df_item['title'])
    print('Dimension of feature vector: {}'.format(tfidf_matrix.shape))
    
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return sim_matrix, df

def content_based_filtering(df, k=20, features=700):
    """
        Generate top-k list according to cosine similarity
    """

    sim_matrix, df = content_processing(df, features)

    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
    df = df[['userId', 'tid', 'title']]

    pred, actual = [], []
    indexes = []
    puid, ptid1, ptid2 = None, None, None
    for index, row in df.iterrows():
        uid, tid = row['userId'], row['tid']

        if uid != puid and puid != None:
            sim_scores = list(enumerate(sim_matrix[ptid1]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:k+1]
            sim_scores = [i for i,j in sim_scores]
            pred.append(sim_scores)
            actual.append(ptid2)
            puid, ptid1, ptid2 = uid, tid, tid

            indexes.append(ptid1)
        else:
            ptid1 = ptid2
            ptid2 = tid
            puid = uid
    
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(sim_matrix[indexes])
    distances, indices = nbrs.kneighbors(sim_matrix[indexes])

    print('\nEvaluation for top-k list:')
    evaluate(pred, actual, k)

    print('\nEvaluation for KNN:')
    evaluate(indices.tolist(), actual, k)