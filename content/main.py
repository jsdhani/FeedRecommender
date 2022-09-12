import os
from pandas import read_pickle, DataFrame, concat
from FeedRecommender.common.constants import MERGED_TEXTS, \
    CONTENT, DESCRIPTION, CAPTION, TITLE, POST_ID
from FeedRecommender.content.fetch_attributes import FetchAttributes
from FeedRecommender.content.merge_attributes import MergeAttributes
from FeedRecommender.content.prepare_cluster_labels import PrepareClusterLabels
from FeedRecommender.content.prepare_feature_set import PrepareFeatureSet
from FeedRecommender.content.process_attributes import ProcessAttributes

class contentProfile:
    """
    This the class that orchestrates the creation of content
    profile attributes that shall be utilized in the downstream
    tasks of unsupervised learning, i.e Clustering of contents.
    The motivation behind clustering is to group contents on a
    criteria not solely based on their category and that also
    utilized their textual information.
    The entire pipeline involves multiple subcomponents, which
    during the process further add to the attribute set
    (such as content source, text language etc.)
    In order to execute this script , use the following main():
    if __name__ == '__main__':
        contentProfile.create_profile()
    """

    @staticmethod
    def fetch_and_merge_attributes():
        """
        Fetch appropriate attributes and
        merge texts to prepare the attribute set
        :return: None, the result is saved in directory
        """
        fa = FetchAttributes(
            data_path=os.getcwd() + '/../data/raw/reacted_posts.json'
        )
        fa.controller()
        tma = MergeAttributes(data=fa.data)
        tma.controller()
        tma.data.to_pickle("intermediates/merged_content.pkl")
    
    @staticmethod
    def process_attributes():
        """
        To process the fetched and merged
        attributes. The process returns the english text
        records separately from the rest of the language records.
        This method includes basic preprocessing techniques and
        vector generation of texts in records
        :return: None, results are stored in directory
        """
        data = read_pickle("intermediates/merged_content.pkl")
        pa = ProcessAttributes(
            data=data,
            process_features=[MERGED_TEXTS]
        )
        en_data, other_data = pa.controller()
        en_data.to_pickle("intermediates/en_data.pkl")
        other_data.to_pickle("intermediates/non_en_data.pkl")

    @staticmethod
    def prepare_feature_set():
        """
        Since the processing of attribute set split the
        data into English and Non-English records, the
        feature set preparation runs once for both of the
        data. Once the results are obtained the results are
         combined and then reformatted to obtain the final
         feature set. Since the number of features are very
         large due to one-hot encoding of attributes,
         dimensionality reduction is applied over the
         feature set so as to make it suitable for
         downstream tasks.
        :return: None, saves the results in directory
        """

        # Working on English data
        en_data = read_pickle("intermediates/en_data.pkl")
        en_pfs = PrepareFeatureSet(
            data=en_data,
            to_drop=[CONTENT, DESCRIPTION,
                     CAPTION, TITLE, MERGED_TEXTS])
        en_pfs.controller(is_en=True)

        # Working on non-English data
        non_en_data = read_pickle("intermediates/non_en_data.pkl")
        non_en_pfs = PrepareFeatureSet(
            data=non_en_data,
            to_drop=[CONTENT, DESCRIPTION,
                     CAPTION, TITLE, MERGED_TEXTS])
        non_en_pfs.controller(is_en=False)

        # Merging the results
        all_data = concat(
            [en_data, non_en_data],
            axis=0).reset_index(drop=True)

        # Preparing the reformatted results
        all_data_vectors = concat(
            [en_pfs.data, non_en_pfs.data],
            axis=0).reset_index(drop=True)

        all_data_vectors = all_data_vectors.fillna(0)

        # Performing dimensionality reduction
        # on the reformatted feature set
        all_data_vectors_reduced = DataFrame(
            en_pfs.get_best_N_components(
                attributes=all_data_vectors.drop(columns=[POST_ID])
            ))

        # Saving the results in directory
        all_data.to_pickle("intermediates/all_data.pkl")
        all_data_vectors.to_pickle("intermediates/all_data_vectors.pkl")
        all_data_vectors_reduced.to_pickle("intermediates/all_data_vectors_reduced.pkl")

    @staticmethod
    def prepare_cluster_labels():
        """
        Apply KMeans clustering over the input feature set
        :return: None, saves the results in directory
        """
        pcl = PrepareClusterLabels(
            original_data=read_pickle("intermediates/all_data.pkl"),
            features=read_pickle("intermediates/all_data_vectors_reduced.pkl")
        )
        pcl.controller()
        pcl.original_data.to_pickle("intermediates/all_cluster_labels.pkl")

    @staticmethod
    def create_profile():
        """
        Driver function for content profile creation.
        :return: None, the results for each step are
        saved in the 'intermediates/' subdirectory
        """
        contentProfile.fetch_and_merge_attributes()
        contentProfile.process_attributes()
        contentProfile.prepare_feature_set()
        contentProfile.prepare_cluster_labels()
