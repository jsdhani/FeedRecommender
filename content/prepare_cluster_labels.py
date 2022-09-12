from pandas import DataFrame, set_option, merge, read_pickle
from sklearn.cluster import KMeans
from FeedRecommender.common.config import CONTENT_CLUSTER_COUNT
from FeedRecommender.common.constants import CLUSTER
set_option("display.max_columns", None)

class PrepareClusterLabels:

    def __init__(
            self,
            original_data: DataFrame,
            features: DataFrame
    ):
        """
        Initialize data member of the class
        :param original_data: dataframe object pandas
        :param features: the feature dataframe object
         to be used for clustering
        """
        self.original_data= original_data
        self.features = features

    def get_cluster_labels(
            self,
            K=CONTENT_CLUSTER_COUNT
    ):
        """
        Get KMeans cluster labels for the inout data
        :param K:
        :return:
        """
        kmeans = KMeans(
            n_clusters=K,
            random_state=42).fit(self.features)

        self.features[CLUSTER] = \
            kmeans.predict(self.features)

    def controller(self):
        """
        Driver function to getting the cluster
        labels for input feature set
        :return:
        """
        self.get_cluster_labels()
        self.original_data = merge(
            self.original_data,
            self.features[CLUSTER],
            left_index=True,
            right_index=True
        )

# if __name__ == '__main__':
#     pcl = PrepareClusterLabels(
#         original_data=read_pickle("intermediates/all_data.pkl"),
#         features=read_pickle("intermediates/all_data_vectors_reduced.pkl")
#     )
#     pcl.controller()
#     pcl.original_data.to_pickle("intermediates/all_cluster_labels.pkl")
#     print(pcl.original_data)