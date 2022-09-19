import os
from pandas import DataFrame, read_pickle
from FeedRecommender.common.constants import POST_ID, ML_LANGUAGE, \
    ML_INTERESTS, SOURCE, TEXT_LANGUAGE, CLUSTER

class PreFormatting:
    """
    This class includes the methods to fetch the results
    obtained from content profile and user profile generation
    procedures. No new information is appended to the existing
    values. These methods include basic processing and
    re-arrangement of attributes.

    The following components are prepared as part of this process:

    ---- 1) Content Vectors: includes attributes only
    related to text embeddings.

    ---- 2) Content Categorical Attributes: include content language,
    category, cluster and other categorical attributes in
    exploded format, i.e. all attributes with multiple values
    per content have a distinct record for each value.

    ---- 3) User Attributes: No formatting is done over
    these attributes. They are saved as is.

    The following main function can be used in order to run the
    controller function of this class:

    if __name__ == '__main__':
    content_vectors = read_pickle(
    os.getcwd() + "/../content/intermediates/all_data_vectors.pkl")
    content_clusters = read_pickle(
    os.getcwd() + "/../content/intermediates/all_cluster_labels.pkl")
    user_vectors = read_pickle(
    os.getcwd() + "/../user/intermediates/input_features.pkl")
    user_interaction = read_pickle(
    os.getcwd() + "/../user/intermediates/user_interaction.pkl"
    )
    pf = PreFormatting(
        content_vectors=content_vectors,
        content_clusters=content_clusters,
        user_vectors=user_vectors,
        user_interaction=user_interaction
    )
    pf.controller()
    """

    def __init__(
            self,
            content_vectors: DataFrame,
            content_clusters: DataFrame,
            user_vectors: DataFrame,
            user_interaction: DataFrame
    ):
        """
        Initialize data members of the class.
        :param content_vectors: dataframe object pandas
        :param content_clusters: dataframe object pandas
        :param user_vectors: dataframe object pandas
        :param user_interaction: dataframe object pandas
        """
        self.content_vectors = content_vectors
        self.content_clusters = content_clusters
        self.user_vectors = user_vectors
        self.user_interaction = user_interaction
        self.user_genres = DataFrame()

    def get_content_text_attributes(self):
        """
        From the complete set of content attributes,
        filter out the ones that represent the text
        embeddings. The function returns the feature
        names as a list of values
        :return: List of text attribute names
        """
        attributes = [POST_ID]
        for column in self.content_vectors.columns:
            if isinstance(column, int):
                attributes.append(column)
        return attributes

    def get_content_features(self):
        """
        This function filters out to keep only the
        categorical attributes. The original content
        features can possible hold multiple values.
        While this representation is concise, exploding
        such attributes will reduce the processing
        when generating the recommendations.
        :return: None, updates the data member
        of the class
        """
        #filtering to keep categorical attributes
        self.content_clusters = self.content_clusters[
            [POST_ID, ML_LANGUAGE, ML_INTERESTS,
             SOURCE, TEXT_LANGUAGE, CLUSTER]]

        # exploding the required attributes
        self.content_clusters = \
            self.content_clusters.explode(ML_LANGUAGE)

        self.content_clusters = \
            self.content_clusters.explode(ML_INTERESTS)

        self.content_clusters = \
            self.content_clusters.reset_index(drop=True)

    def save_results(self):
        """
        Saving the results in directory for future
        reference during recommendation
        :return: None, the results are saved in the directory
        """
        self.content_clusters.to_pickle("data/content_cat.pkl")
        self.content_vectors.to_pickle("data/content_text_vec.pkl")
        self.user_vectors.to_pickle("data/users.pkl")
        self.user_interaction.to_pickle("data/user_interaction.pkl")

    def get_user_features(self):
        pass

    def controller(self):
        """
        Driver function to process and save content and user
        specific attributes to be utilized during
        recommendation results preparation
        :return: None, the results are saved in the directory
        """
        self.content_vectors = self.content_vectors[
            self.get_content_text_attributes()
        ]
        self.get_content_features()
        self.get_user_features()
        self.save_results()
