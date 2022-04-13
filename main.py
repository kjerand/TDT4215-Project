from cbf import content_based_filtering
from cb import collaborative_filtering_svd
from utils import load_data, cbf_plot_no_of_feature, plot_svd

if __name__ == "__main__":
    df = load_data("active1000")
   
    print("\nRecommendation based on collaborative filtering with SVD...")
    collaborative_filtering_svd(df)
    

    print("\nRecommendation based on content based filtering with top-k and KNN...")
    
    cbf_plot_no_of_feature(df, knn=False)
    cbf_plot_no_of_feature(df, knn=True)

    content_based_filtering(df, knn=False, features=700, print_result=True)
    content_based_filtering(df, knn=True, features=100, print_result=True)