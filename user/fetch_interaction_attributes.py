import json
from pandas import DataFrame, read_pickle, set_option, get_dummies
from FeedRecommender.common.constants import REACTIONS, ML_LANGUAGE, \
    ML_INTERESTS, SOURCE, TEXT_LANGUAGE, POST_ID, \
    CONTENT, DESCRIPTION, CAPTION, TITLE, MERGED_TEXTS, CLUSTER
set_option("display.max_columns", None)

class FetchInteractionAttributes:

    def __init__(
            self,
            interaction_path: str,
            content_path: str
    ):
        """
        Fetch required data from path to
        initialize the data members
        :param interaction_path: string value path to file
        :param content_path: string value path to file
        """
        with open(interaction_path) as json_file:
            self.interaction_data = DataFrame(json.load(json_file))

        self.content_data = read_pickle(content_path)
        self.clusters = self.content_data[[POST_ID, CLUSTER]]
        self.content_data.drop(columns=[CLUSTER], inplace=True)

    def get_posts(
            self,
            reactions: list
    ) -> list:
        """
        Retrieve post_id from the key value pairs
        :param reactions: list key-value pairs
        :return: list of post_ids
        """
        return [post[POST_ID] for post in reactions]

    def explode_attribute(
            self,
            data: DataFrame,
            feature: str,
            dropna: bool = True
    ):
        """
        Split dataframe attribute consisting of list values
        into separate records each with single value
        :param data: dataframe object pandas
        :param feature: feature to be splitted
        :param dropna: if True drop records consisting of NaN
        :return: dataframe object pandas
        """
        data = data.explode(feature).reset_index(drop=True)
        if dropna:
            data = data.dropna().reset_index(drop=True)
        return data

    def filter_attributes(
            self,
            to_drop: list
    ):
        """
        Filter to drop the unnecessary attributes
        :param to_drop: list of unnecessary attributes
        :return: None, updates the data member of the class
        """
        self.content_data = self.content_data.drop(
            columns=to_drop
        ).reset_index(drop=True)

    def encode_attributes(
            self,
            features,
            data: DataFrame
    ) -> DataFrame:
        """
        Generate One-Hot encoded attributes out of a
        single attribute
        :param features: the features to be encoded
        :param data: dataframe object pandas
        :return: dataframe object pandas
        """
        return get_dummies(data=data, columns=features)

    def prepare_interaction_data(self):
        """
        Prepare user-post interaction data attributes
        :return: None, updates the data member of the class
        """
        self.interaction_data[REACTIONS] = \
            [self.get_posts(reactions)
             for reactions in self.interaction_data[REACTIONS]]

        self.interaction_data = self.explode_attribute(
            feature=REACTIONS,
            data=self.interaction_data
        )

    def prepare_content_data(self):
        """
        Prepare content information data attributes.
        The process includes the following sub-procedures:
        1) Exploding attributes with a list of values in each record
        2) One-hot encoding categorical attributes
        3) Aggregating attributes to represent a single record per content
        :return: None, updates the data member of the class
        """
        self.filter_attributes(
            to_drop=[CONTENT, DESCRIPTION, CAPTION,
                     TITLE, MERGED_TEXTS, ML_LANGUAGE,
                     SOURCE, TEXT_LANGUAGE]
        )

        #Exploding attributes with a list of values in each record
        self.content_data = self.explode_attribute(
            data=self.content_data,
            feature=ML_INTERESTS,
            dropna=False
        )

        #One-hot encoding categorical attributes
        self.content_data = self.encode_attributes(
            features=[ML_INTERESTS],
            data=self.content_data
        )

        # Aggregating attributes to represent a single
        # record per content
        attributes = self.content_data.columns.tolist()
        attributes.remove(POST_ID)
        self.content_data = self.content_data.\
            groupby(POST_ID).sum().reset_index()

        for attribute in attributes:
            self.content_data[attribute].values[
                self.content_data[attribute].values > 0] = 1

    def controller(self):
        """
        Driver function to generate content information attributes
        and user-content interaction attributes
        to be used in downstream user profile creation
        :return: None, updates the data member of the class
        """
        self.prepare_interaction_data()
        self.prepare_content_data()
