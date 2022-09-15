from pandas import read_pickle, get_dummies
from FeedRecommender.common.constants import LANGUAGE, USER_ID

class FetchUserInfoAttributes:

    def __init__(
            self,
            data_path: str
    ):
        """
        Fetch required data from path to
        initialize the data members
        :param data_path: string value path to file
        """
        self.data = read_pickle(data_path)

    def explode_attribute(
            self,
            feature: str
    ):
        """
        Split dataframe attribute consisting of list values
        into separate records each with single value
        :param feature: feature consisting of list values
        to be split
        :return: None, updates the data member of the class
        """
        self.data = self.data.explode(feature).\
            reset_index(drop=True)

    def encode_language(self):
        """
        Generate One-Hot encoded attributes out of a
        single attribute
        :return: None, updates the data member of the class
        """
        self.data = get_dummies(
            data=self.data,
            columns=[LANGUAGE]
        )

    def controller(self):
        """
        Driver function to generate user information attributes
        to be used in downstream user profile creation
        :return: None, updates the data member of the class
        """
        self.explode_attribute(feature=LANGUAGE)
        self.encode_language()
        self.data = self.data.groupby(USER_ID).sum().reset_index()
        attributes = self.data.columns.tolist()
        attributes.remove(USER_ID)

        for attribute in attributes:
            self.data[attribute].values[
                self.data[attribute].values > 0] = 1
