from FeedRecommender.common.constants import REACTIONS, \
    TOTAL_VIEWS, AGE, SCORES
from FeedRecommender.recsys.common import FetchData
from pandas import DataFrame, merge

class GetTrendingScores:

    @staticmethod
    def get_total_views(user_interaction: DataFrame):
        """
        Calculate the total number of views for each
        content in user interaction data
        :param user_interaction: dataframe object pandas
        :return: dataframe object pandas
        """
        total_views = user_interaction.groupby(REACTIONS).size().reset_index()
        total_views.rename(columns={0: TOTAL_VIEWS}, inplace=True)
        return total_views

    @staticmethod
    def get_age(user_interaction: DataFrame):
        """
        Compute the age of each content in user interaction data.
        In order to calculate the age, it has been assumed that
        each record was stored at a different timestamp
        :param user_interaction: dataframe object pandas
        :return: dataframe object pandas
        """
        user_interaction[AGE] = user_interaction.index
        return user_interaction.groupby(REACTIONS)[AGE].max().reset_index()

    @staticmethod
    def get_trending_score(contents: DataFrame):
        """
        Calculate trending score using total views and age
        :param contents: dataframe object pandas
        :return: dataframe object pandas with trending scores
        """
        contents[SCORES] = contents[TOTAL_VIEWS] / contents[AGE] ** 0.2
        return contents

    @staticmethod
    def controller(save_results: bool = False):
        """
        Driver function to generate trending scores for
        each content in user interaction data
        :param save_results: boolean indicator.
        If True, the results will be stored in
        directory else not
        :return: None, the result is saved in directory
        """
        log = FetchData.get_user_interaction()
        total_views = GetTrendingScores.get_total_views(user_interaction=log)
        age = GetTrendingScores.get_age(user_interaction=log)
        content_stats = merge(total_views, age, on=REACTIONS)
        content_stats = GetTrendingScores.get_trending_score(contents=content_stats)
        content_stats = content_stats.sort_values(by=SCORES, ascending=False).\
            reset_index(drop=True)
        if save_results:
            content_stats.to_pickle("data/trending.pkl")
