import numpy as np
from pandas import DataFrame, get_dummies, set_option, \
    read_pickle, merge,  concat
from numpy import asarray
from sklearn.decomposition import PCA
import spacy
from FeedRecommender.common.config import PCA_DIMENSIONS, \
    SPACY_VECTOR_LENGTH
from FeedRecommender.common.constants import ML_LANGUAGE, \
    TEXT_LANGUAGE, SOURCE, ML_INTERESTS, CONTENT, DESCRIPTION, \
    CAPTION, TITLE, MERGED_TEXTS, POST_ID
nlp = spacy.load("en_core_web_lg")
set_option("display.max_columns", None)

class PrepareFeatureSet:

    def __init__(
            self,
            data: DataFrame,
            to_drop: list
    ):
        """
        Initialize data members of the class
        :param data: dataframe object pandas
        :param to_drop: list of attributes to be dropped
        """
        self.data = data
        self.to_drop = to_drop

    def filter_attributes(self):
        """
        Dropping out the attributes not required for
        further procedures
        :return: None, updates the data member of the class
        """
        self.data = self.data.drop(
            columns=self.to_drop).reset_index(drop=True)

    def explode_attribute(self, feature):
        """
        Split dataframe attribute consisting of list values
        into separate records each with single value
        :param feature: feature consisting of list values
        to be split
        :return: None, updates the data member of the class
        """
        self.data = self.data.explode(feature).\
            reset_index(drop=True)

    def encode_attributes(self, features):
        """
        Generate One-Hot encoded attributes out of a
        single attribute
        :param features: feature to preform one-hot encoding on
        :return: None, updates the data member of the class
        """
        self.data = get_dummies(data=self.data,
                                columns=features)

    def get_word_vectors(
            self,
            feature: str,
            is_en: bool
    ):
        """
        Generate vector representation of list of texts.
        If the language of texts is English, generate vector
        representation, else return list of zero vectors
        :param feature: dataframe attribute with list of texts
         to work on
        :param is_en: boolean indicator. If true, text language is
        English else not
        :return: vectorized format of text
        """
        if is_en:
            return asarray([nlp(text).vector
                            for text in self.data[feature]])
        else:
            return [np.zeros(SPACY_VECTOR_LENGTH)
                    for _ in self.data[feature]]

    def get_best_N_components(
            self,
            attributes,
            N=PCA_DIMENSIONS
    ):
        """
        Perform dimensionality reduction using PCA
        :param attributes: original attribute set
        :param N: number of attributes to reduce down to
        :return: Transformed attribute set with reduced attributes
        """
        pca = PCA(n_components=N)
        return pca.fit_transform(attributes).tolist()

    def controller(
            self,
            is_en: bool
    ):
        """
        Driver function to prepare the feature set for
        downstream clustering tasks
        :param is_en: boolean indicator. If true, text language is
        English else not
        :return: None, updates the data member of the class
        """
        vectors = DataFrame(self.get_word_vectors(
            feature=MERGED_TEXTS, is_en=is_en))
        vectors[POST_ID] = self.data[POST_ID]

        self.filter_attributes()
        self.explode_attribute(feature=ML_LANGUAGE)
        self.explode_attribute(feature=ML_INTERESTS)
        self.explode_attribute(feature=SOURCE)
        self.encode_attributes(features=[ML_INTERESTS, ML_LANGUAGE,
                                         SOURCE, TEXT_LANGUAGE])

        attributes = self.data.columns.tolist()
        attributes.remove(POST_ID)
        self.data = self.data.groupby(POST_ID).sum().reset_index()
        for attribute in attributes:
            self.data[attribute].values[
                self.data[attribute].values > 0] = 1

        self.data = merge(self.data, vectors, on=POST_ID)
