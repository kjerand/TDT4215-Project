from cbf import content_based_filtering
from cb import collaborative_filtering_svd
from utils import load_data

if __name__ == "__main__":
    df = load_data("active1000")
   
    print("\nRecommendation based on collaborative filtering with SVD...")
    collaborative_filtering_svd(df)

    print("\nRecommendation based on content based filtering with top-k and KNN...")
    content_based_filtering(df)