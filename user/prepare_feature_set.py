from pandas import DataFrame, merge, set_option
from FeedRecommender.common.config import CONTENT_CLUSTER_COUNT
from FeedRecommender.common.constants import POST_ID, CLUSTER, REACTIONS, USER_ID
from numpy import asarray, zeros, arange
from tqdm import tqdm
set_option("display.max_columns", None)

class PrepareUserFeatureSet:

    def __init__(
            self,
            interaction_data: DataFrame,
            content_data: DataFrame,
            user_info: DataFrame,
            clusters: DataFrame
    ):
        """
        Initializing the data members
        :param interaction_data: user content interaction
        dataframe object pandas
        :param content_data: content information dataframe
        object pandas
        :param user_info: user information dataframe
        object pandas
        :param clusters: content cluster labels dataframe
        object pandas
        """
        self.interaction_data = interaction_data
        self.content_data = content_data
        self.user_info = user_info
        self.clusters = clusters[[POST_ID, CLUSTER]]

    def get_index_encoded_vector(
            self,
            df: DataFrame
    ) -> list:
        """
        Convert the cluster labels from a list of cluster
        labels to a target variable format suitable for input
        to multi-label classification model. The updated format
        consists of a fixed length vector of K (number of clusters)
        elements with value set to one for every value
        in the original list
        :param df: dataframe object pandas
        :return: list of encoded labels
        """
        encoded_labels = []

        for index in tqdm(range(len(df))):
            clusters = asarray(df.loc[index, CLUSTER])
            encoded_clusters = zeros(
                (clusters.size, CONTENT_CLUSTER_COUNT)
            )
            encoded_clusters[arange(clusters.size), clusters] = 1

            if len(encoded_clusters) == 1:
                encoded_labels.append(encoded_clusters[0].tolist())
                continue

            encoded_vector = encoded_clusters.sum(axis=0).tolist()
            encoded_vector = [1 if element > 0 else 0
                              for element in encoded_vector]
            encoded_labels.append(encoded_vector)

        return encoded_labels

    def prepare_Y(self) -> DataFrame:
        """
        Prepare the target variable in appropriate
        multi-label classification format
        :return: dataframe object pandas
        """
        Y = merge(self.interaction_data,
                  self.clusters,
                  left_on=REACTIONS,
                  right_on=POST_ID)
        Y = Y.groupby(USER_ID)[CLUSTER].apply(list).reset_index()
        encoded_labels = self.get_index_encoded_vector(df=Y)
        Y[CLUSTER] = encoded_labels
        return Y

    def prepare_X(self) -> DataFrame:
        """
        Prepare the independent attribute set by merging
        and aggregating user information and content
        information attributes
        :return: dataframe object pandas
        """
        X = merge(self.user_info,
                  self.interaction_data,
                  on=USER_ID, how="left")
        X = merge(X,
                  self.content_data,
                  left_on=REACTIONS,
                  right_on=POST_ID, how="left")

        X = X.fillna(0)
        X.drop(columns=[POST_ID, REACTIONS], inplace=True)
        X = X.groupby(USER_ID).sum().reset_index()

        attributes = X.columns.tolist()
        attributes.remove(USER_ID)
        for attribute in attributes:
            X[attribute].values[X[attribute].values > 0] = 1

        return X

    def controller(self) -> DataFrame:
        """
        Driver function to create input data attributes
        and target attribute for the downstream task
        of multi-label classification
        :return: dataframe object pandas
        """
        X = self.prepare_X()
        Y = self.prepare_Y()
        print(X.shape)
        print(Y.shape)
        merged = merge(X, Y,
                       on=USER_ID, how="left")
        merged = merged.fillna(-1)
        merged[CLUSTER] = [[0] * CONTENT_CLUSTER_COUNT
                           if vector == -1 else vector
                           for vector in merged[CLUSTER]]
        return merged
