from pandas import DataFrame
from FeedRecommender.common.config import RECSYS_HISTORY_CAP
from FeedRecommender.common.constants import REC_TYPE, POST_ID, REACTIONS


class GetTrendingRecs:

    @staticmethod
    def controller(
            trending: DataFrame,
            result: DataFrame,
            upper_cap: int = RECSYS_HISTORY_CAP
    ) -> DataFrame:
        """
        Return the top trending recommendations based
        on the pre-computed trending scores
        :param trending: dataframe object pandas of
        content-wise trending scores
        :param result: dataframe object pandas of
        already prepared result
        :param upper_cap: maximum number of
        recommendations to be prepared of this type
        :return: dataframe object pandas
        """
        rec_type = "TRENDING"

        trending[REC_TYPE] = rec_type
        trending = trending[[POST_ID, REC_TYPE]]

        # drop the contents already included
        # in the result to avoid duplicates
        if result is not None:
            trending = trending[
                ~trending[POST_ID].isin(result[POST_ID])
            ].reset_index(drop=True)

        return trending \
            if upper_cap is None \
            else trending.loc[:upper_cap, :]
