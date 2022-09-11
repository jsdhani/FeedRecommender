import json
from pandas import DataFrame, set_option
from tqdm import tqdm
from FeedRecommender.common.constants import \
    DESCRIPTION, CAPTION, TITLE, CONTENT, SOURCE, POST_ID
set_option("display.max_columns", None)

class FetchAttributes:

    def __init__(
            self,
            data_path: str
    ):
        """
        Retrieve data from the string data path to
        assign to initialize the data member
        :param data_path: string valued data path to file
        """
        with open(data_path) as json_file:
            self.data = DataFrame(json.load(json_file))

    def check_attribute_null_values(
            self,
            attribute: str
    ) -> bool:
        """
        Check for any potential NaN values in the
        records of a particular attribute
        :param attribute: dataframe object attribute
        :return: Boolean indicator, if True, NaN
        values exist else not
        """
        return self.data[attribute].isnull().values.any()

    def split_content_attribute(self):
        """
        Split the key-value pairs comprising of any
        combination of caption, title, description
        into 3 distinct attributes for convenience
        :return: None, updates the data member of the class
        """
        for index in tqdm(range(len(self.data))):
            content = self.data.loc[index, CONTENT]

            if DESCRIPTION in content.keys():
                self.data.loc[index, DESCRIPTION] = content[DESCRIPTION]

            if CAPTION in content.keys():
                self.data.loc[index, CAPTION] = content[CAPTION]

            if TITLE in content.keys():
                self.data.loc[index, TITLE] = content[TITLE]

    def split_post_attribute(self):
        """
        Acquire source information from the post_id
        attribute in dataframe
        :return: None, updates the data member of the class
        by adding a new attribute
        """
        self.data[SOURCE] = [post_id.split("_")[0]
                             for post_id in self.data[POST_ID]]

    def fill_empty_values(
            self,
            value=""
    ):
        """
        Fill NaN cell records with specified default value
        :param value: value to replace null with
        :return: None, updates the data member of the class
        """
        self.data = self.data.fillna(value)

    def controller(self):
        """
        Driver function to fetch the content
        information attributes
        :return: None, updates the data member of the class
        """
        self.split_content_attribute()
        self.split_post_attribute()
        self.fill_empty_values()
