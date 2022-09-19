import os
from typing import Any
from pandas import read_pickle, DataFrame
from FeedRecommender.user.fetch_info_attributes import FetchUserInfoAttributes
from FeedRecommender.user.fetch_interaction_attributes import FetchInteractionAttributes
from FeedRecommender.user.prepare_feature_set import PrepareUserFeatureSet

class userProfile:
    """
    The main class to execute pipeline to create user
    profile specific attributes.
    This class includes all static methods.
    In order to obtain the results, use the following
    main function:
    if __name__ == '__main__':
        user_profile = userProfile.create_profile(save_results=False)
    """

    @staticmethod
    def get_user_info():
        """
        Create instance to prepare user information
        attributes
        :return: user_info object
        """
        return FetchUserInfoAttributes(
            data_path=os.getcwd() +
            "/../data/complete_user_info.pkl"
        )

    @staticmethod
    def get_user_interaction():
        """
        Create instance to prepare user interaction
        attributes
        :return: user_interactions object
        """
        return FetchInteractionAttributes(
            interaction_path=os.getcwd() +
            "/../data/raw/user_post_interaction.json",

            content_path=os.getcwd() +
            "/../content/intermediates/all_cluster_labels.pkl"
        )

    @staticmethod
    def get_feature_set(
            user_info: Any,
            user_interaction: Any
    ):
        """
        Create instance to consolidate all user_info,
        user interaction and content info attributes
        :return: user feature set object
        """
        return PrepareUserFeatureSet(
            interaction_data=user_interaction.interaction_data,
            content_data=user_interaction.content_data,
            user_info=user_info.data,
            clusters=read_pickle(
                os.getcwd() +
                "/../content/intermediates/all_cluster_labels.pkl")
        )

    @staticmethod
    def create_profile(
            save_results: bool = True
    ) -> DataFrame:
        """
        Driver function to create user profile attributes to
        be used for downstream supervised neural model training.
        The attribute set comprises of 3 sub-components:
        1) User Info: contain information about user's
        language preferences
        2) User Interaction: contain information about each
        posts_id viewed by every user
        3) Content Info: attributes particular to contents,
        including the obtained clustering results
        :param save_results: Boolean Indicator to check whether
        to store results or not
        :return: The consolidated user profile dataframe
        object pandas
        """

        #preparing the User Info attributes
        user_info = userProfile.get_user_info()
        user_info.controller()

        #preparing user interaction and content info attributes
        user_interaction = userProfile.get_user_interaction()
        user_interaction.controller()

        #consolidating all acquired attributes into a
        # single user profile
        user_features = userProfile.get_feature_set(
            user_info=user_info,
            user_interaction=user_interaction
        )
        user_profile = user_features.controller()

        if save_results:
            user_profile.to_pickle(
                "intermediates/input_features.pkl")
            user_interaction.interaction_data.to_pickle(
                "intermediates/user_interaction.pkl"
            )

        return user_profile
