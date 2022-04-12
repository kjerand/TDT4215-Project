
import numpy as np
from scipy.linalg import sqrtm
from utils import load_dataset
from evaluate import rmse
from sklearn.metrics import mean_squared_error, mean_absolute_error


def svd(train, k):
    utility_matrix = np.array(train)    # the nan or unavailable entries are masked
    mask = np.isnan(utility_matrix)
    masked_arr = np.ma.masked_array(utility_matrix, mask)
    item_means = np.mean(masked_arr, axis=0)    # nan entries will replaced by the average rating for each item
    utility_matrix = masked_arr.filled(item_means)    
    means_array = np.tile(item_means, (utility_matrix.shape[0],1))    # we remove the per item average from all entries.
    # the above mentioned nan entries will be essentially zero now
    utility_matrix = utility_matrix - means_array    # The magic happens here. U and V are user and item features
    U, s, V = np.linalg.svd(utility_matrix, full_matrices=False)
    s = np.diag(s)    # we take only the k most significant features
    s = s[0:k,0:k]
    U = U[:,0:k]
    V = V[0:k,:]    
    s_root = sqrtm(s)    
    Usk = np.dot(U,s_root)
    skV = np.dot(s_root,V)
    UsV = np.dot(Usk, skV)    
    UsV = UsV + means_array    

    return UsV


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

def collaborative_filtering_svd(df):
    ratings = load_dataset(df)
    no_of_features = [2]

    train, test = train_test_split(ratings, fraction=0.2)

    users, items = df['userId'].unique(), df['documentId'].unique()

    users_amount, items_amount = ratings.shape

    print(users_amount, items_amount)

    users_index = {users[i]: i for i in range(users_amount)}
    items_index = {items[i]: i for i in range(items_amount)}

    for f in no_of_features: 
        svdout = svd(train, f)

        pred = [] #to store the predicted ratings    
        actual = []
        for _,row in df.iterrows():
            user = row['userId']
            item = row['documentId']        
            u_index = users_index[user]
            if item in items_index:
                i_index = items_index[item]
                pred_rating = svdout[u_index, i_index]
                actual_rating = test[u_index, i_index]
                
            else:
                pred_rating = np.mean(svdout[u_index, :])
                
            pred.append(pred_rating)
            actual.append(actual_rating)


        print("RMSE: " , rmse(actual, pred))
        print("MSE: " , mean_squared_error(actual, pred, squared = False ))
        print("MAE: ", mean_absolute_error(actual, pred))