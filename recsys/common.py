import os
from pandas import read_pickle, DataFrame
import tensorflow as tf
from FeedRecommender.common.config import USER_CONTENT_MODEL_NAME
from FeedRecommender.common.constants import CLUSTER, USER_ID

class FetchData:

    @staticmethod
    def get_content_embeddings():
        """
        Return the content text embeddings attributes
        :return: dataframe object pandas
        """
        return read_pickle("data/content_text_vec.pkl")

    @staticmethod
    def get_content_attributes():
        """
        Get the complete set of content categorical attributes
        :return: dataframe object pandas
        """
        return read_pickle("data/content_cat.pkl")

    @staticmethod
    def get_user_attributes():
        """
        Get the complete set of user attributes
        :return: dataframe object pandas
        """
        return read_pickle("data/users.pkl")

    @staticmethod
    def get_user_language():
        """
        Return the user language preferences data
        :return: dataframe object pandas
        """
        return read_pickle("data/complete_user_info.pkl")

    @staticmethod
    def get_user_interaction():
        """
        Return the user-content interaction data
        :return: dataframe object pandas
        """
        return read_pickle("data/user_interaction.pkl")

    @staticmethod
    def get_user_genres():
        """
        Return the genre preferences of all the users
        :return: dataframe object pandas
        """
        return read_pickle("data/user_genres.pkl")

    @staticmethod
    def get_trending():
        """
        Return contents sorted by their decreasing
        order of trending scores
        :return: dataframe object pandas
        """
        return read_pickle("data/trending.pkl")

    @staticmethod
    def get_user_content_cluster_model():
        """
        Load multi-label classification neural model
        :return: saved model
        """
        return tf.keras.models.\
            load_model(os.getcwd() + "/../user/model/" +
                       USER_CONTENT_MODEL_NAME)

    @staticmethod
    def get_contents_for_cluster(
            contents: DataFrame,
            clusters: list
    ) -> DataFrame:
        """
        For a given list of clusters, return the
        contents belonging to those clusters
        :param contents: dataframe object pandas
        :param clusters: list of clusters shortlisted
        :return: dataframe object pandas
        """
        return contents[
            contents[CLUSTER].isin(clusters)].\
            reset_index(drop=True)

    @staticmethod
    def get_previously_viewed(
            user_interaction: DataFrame,
            user_id: str
    ) -> DataFrame:
        """
        Return the contents previously viewed by a user
        :param user_interaction: dataframe object pandas
        :param user_id: string value for user id
        :return: dataframe object pandas
        """
        return user_interaction[
            user_interaction[USER_ID] == user_id].\
            reset_index(drop=True)
