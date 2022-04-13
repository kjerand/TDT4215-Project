from cbf import content_based_filtering
from cb import collaborative_filtering_svd
from utils import load_data, cbf_plot_no_of_feature, plot_svd

if __name__ == "__main__":
    df = load_data("active1000")
   
    print("\nRecommendation based on collaborative filtering with SVD...")
    features = [2, 4, 6]
    rmse, mse, mae = collaborative_filtering_svd(df, features)
    plot_svd(features, rmse, mse, mae)

    print("\nRecommendation based on content based filtering with top-k and KNN...")
    df = df[0:250000]
    content_based_filtering(df)

    cbf_plot_no_of_feature(df, knn=False)
    cbf_plot_no_of_feature(df, knn=True)