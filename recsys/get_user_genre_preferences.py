from FeedRecommender.common.constants import USER_ID, ML_INTERESTS, ID, GENRE_PREFERENCES
from FeedRecommender.recsys.common import FetchData
from pandas import DataFrame

class GetUserGenrePreferences:

    def __init__(self):
        """
        Initialize data member of the class
        """
        self.users = FetchData.get_user_attributes()

    def get_genre_specific_attributes(self):
        """
        Filter to keep only the genre specific attributes
        from the users dataframe object
        :return: list of genre attributes
        """
        genres = [USER_ID]
        for column in self.users.columns:
            if ML_INTERESTS in column:
                genres.append(column)
        return genres

    def get_user_genres(
            self,
            genres: list
    ) -> DataFrame:
        """
        Format the nomenclature of genre attributes
        :param genres: list of genre attributes
        :return: dataframe object pandas
        """
        user_genres = self.users[genres]
        user_genres.columns = [column.split("_")[-1]
                               for column in user_genres.columns]
        user_genres = user_genres.rename(columns={ID: USER_ID})
        return user_genres

    def format_encoding(
            self,
            user_genres: DataFrame
    ):
        """
        Basic formatting of one-hot encoded genre attributes
        such that all values with 1 are replaced with
        their genre names
        :param user_genres: dataframe object pandas
        :return: dataframe object pandas
        """
        for column in user_genres.columns:
            if column == USER_ID:
                continue
            user_genres[column] = [column if value > 0 else 0
                                   for value in user_genres[column]]
        return user_genres

    def get_genre_preferences(
            self,
            user_genres: DataFrame
    ) -> DataFrame:
        """
        Prepare the genre preference attribute for
        each user in dataframe object pandas
        :param user_genres: dataframe object pandas
        :return: dataframe object pandas with
        user genre attributes
        """
        genre_preferences = []
        for index in range(len(user_genres)):
            vector = user_genres.iloc[index, 1:].values.tolist()
            vector = [val for val in vector if val != 0]
            genre_preferences.append(vector)
        user_genres[GENRE_PREFERENCES] = genre_preferences
        return user_genres[[USER_ID, GENRE_PREFERENCES]]

    def controller(
            self,
            save_result: bool = False
    ):
        """
        Driver function to compute user genre preferences
        :param save_result: boolean indicator.
        If true, save the results in directory else not
        :return: None, the results are saved in directory
        """
        genres = self.get_genre_specific_attributes()
        user_genres = self.get_user_genres(genres=genres)
        user_genres = self.format_encoding(user_genres=user_genres)
        genre_preferences = self.get_genre_preferences(user_genres=user_genres)

        if save_result:
            genre_preferences.to_pickle("data/user_genres.pkl")
