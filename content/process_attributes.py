from FeedRecommender.common.config import UNWANTED_CHAR_WORDS
from FeedRecommender.common.constants import MERGED_TEXTS, EN_LANGUAGE, ML_LANGUAGE
from pandas import DataFrame, set_option, Series, read_pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
from tqdm import tqdm
nlp = spacy.load("en_core_web_lg")
lemmatizer = WordNetLemmatizer()
nltk.download('omw-1.4')
set_option("display.max_columns", None)

class ProcessAttributes:

    def __init__(
            self,
            data: DataFrame,
            process_features: list  # MERGED_TEXTS
    ):
        """
        Initialize data members of the class
        :param data: dataframe object pandas
        :param process_features: list of features
        to be processed
        """
        self.data = data
        self.process_features = process_features

    def to_lower(
            self,
            feature: Series
    ) -> Series:
        """
        Convert string text to lowercase
        :param feature: text to work with
        :return: lowercase format of input text
        """
        return feature.str.lower()

    def remove_unwanted_words(
            self,
            text: str
    ) -> str:
        """
        Remove unwanted words from the input text,
        i.e remove words with unnecessary characters
        that imply irrelevant information in current
        context. For eg: hyperlinks, special
        characters' Unicode representation etc.
        :param text: Text to work with
        :return: text cleaned off unwanted words
        """
        for unwanted_word in UNWANTED_CHAR_WORDS:
            text = ' '.join(
                word for word in text.split()
                if unwanted_word not in word
            )
        return text

    def remove_non_alpha_numerics(
            self,
            text: str
    ) -> str:
        """
        Filter to proceed with only alpha
        numeric characters in text
        :param text: Text to work with
        :return: alpha-numeric string
        """
        return re.sub(r'[^a-zA-Z \']', ' ', text)

    def remove_redundant_whitespaces(
            self,
            text: str
    ) -> str:
        """
        Remove unnecessary whitespaces from the text
        :param text: Text to work with
        :return: Text with no multiple consecutive
        occurrence of whitespace
        """
        text = text.lstrip()
        text = text.rstrip()
        return re.sub("\\s+", " ", text)

    def lemmatize_str(
            self,
            feature: Series
    ) -> Series:
        """
        Perform text lemmatization
        :param feature: list of strings to work with
        :return: lemmatize format of the input list of texts
        """
        feature = [lemmatizer.lemmatize(x)
                   for x in feature.tolist()]
        return Series(feature)

    def process_data(
            self,
            df: DataFrame,
            features: list,
            is_lang_en: bool
    ):
        """
        The preprocessing pipeline is executed via this method.
        :param df: dataframe object pandas
        :param features: list of features to process
        :param is_lang_en: validate whether working with
        english language text
        :return: Dataframe object pandas
        """
        for feature in features:

            df[feature] = self.to_lower(
                feature=df[feature])

            df[feature] = [self.remove_redundant_whitespaces(text=text)
                           for text in df[feature]]

            if not is_lang_en:
                continue

            df[feature] = [self.remove_unwanted_words(text=text)
                           for text in df[feature]]

            df[feature] = [self.remove_non_alpha_numerics(text=text)
                           for text in df[feature]]

            df[feature] = [self.remove_redundant_whitespaces(text=text)
                           for text in df[feature]]

            df[feature] = self.lemmatize_str(feature=df[feature])

        return df

    def get_en_data(self):
        """
        Obtain data records specific to English language.
        Separating English and non-English language records
        helps in conveniently dealing with downstream tasks
        of vectoring natural language for texts in english.
        :return: English data records and rest of the records
        in dataframe object pandas formats
        """
        en_data_indices = []
        non_en_data_indices = []

        for index in tqdm(range(len(self.data))):
            if EN_LANGUAGE in self.data.loc[index, ML_LANGUAGE]:
                en_data_indices.append(index)
                continue
            non_en_data_indices.append(index)

        return self.data.filter(items=en_data_indices,
                                axis=0).reset_index(drop=True),\
            self.data.filter(items=non_en_data_indices,
                             axis=0).reset_index(drop=True)

    def controller(self):
        """
        Driver function to process the fetched and merged
        attributes. The process returns the english text
        records separately from the rest of the language records
        :return: English data records and rest of the records
        in dataframe object pandas formats
        """
        en_data, non_en_data = self.get_en_data()
        en_data = self.process_data(df=en_data,
                                    features=self.process_features,
                                    is_lang_en=True)
        non_en_data = self.process_data(df=non_en_data,
                                        features=self.process_features,
                                        is_lang_en=False)

        return en_data, non_en_data

