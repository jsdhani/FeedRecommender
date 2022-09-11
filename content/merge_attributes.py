import spacy
from tqdm import tqdm
from pandas import DataFrame
from FeedRecommender.common.constants import \
    DESCRIPTION, CAPTION, TITLE, MERGED_TEXTS, TEXT_LANGUAGE
from FeedRecommender.common.config import \
    SEMANTIC_OVERLAP_THRESHOLD
from langdetect import detect, LangDetectException
from FeedRecommender.content.fetch_attributes import FetchAttributes

#python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")

class MergeAttributes:

    def __init__(
            self,
            data: DataFrame
    ):
        """
        Initialize data members of the class
        :param data: dataframe object pandas
        """
        self.data = data
        self.merged_text = []
        self.text_lang = []

    def description_required(
            self,
            caption: str,
            title: str
    ) -> bool:
        """
        Check whether to use description text
        in the absence of caption and title or not
        :param caption: caption string
        :param title: title string
        :return: boolean indicator, if True, use
        description instead of title and caption else vice-versa
        """
        return True if not caption and not title else False

    def detect_lang(
            self,
            text: str
    ) -> str:
        """
        Detect the text language
        :param text: text to detect language of
        :return: language string code. If in case,
        the language detection fails, a LangDetectException
         is raised that returns the language code as "none"
        """
        try:
            return detect(text)
        except LangDetectException:
            return "none"

    def check_semantic_overlap(
            self,
            string1: str,
            string2: str
    ) -> float:
        """
        Check how similar are the captions and titles
        to each other. If the similarity score exceeds
        the pre-defined threshold, proceed only with the
        title, else, consider the concatenation of both
        :param string1: one of the texts to be compared
        :param string2: the other text to be compared
        :return: real-valued similarity score
        """
        return nlp(string1).similarity(nlp(string2))

    def merge_texts(
            self,
            string1: str,
            string2: str
    ) -> str:
        """
        Concatenate strings and return
        :param string1: The first string
        :param string2: The string to be
        concatenated to the first string
        :return: Concatenated string object
        """
        return string1 + "." + string2

    def controller(self):
        """
        Driver function to merge the acquired attributes.
        The procedure involves the following sub-components:
        1) Choose to proceed with description or title+caption
        for each record
        2) Check for semantic overlap between title and caption
        to avoid redundancy.
        3) Create an additional attribute indicating the
        text language
        :return:
        """
        for index in tqdm(range(len(self.data))):

            caption = self.data.loc[index, CAPTION]
            title = self.data.loc[index, TITLE]
            description = self.data.loc[index, DESCRIPTION]

            # Choose to proceed with description or title+caption
            # for each record
            if self.description_required(
                    caption=caption,
                    title=title
            ):
                self.merged_text.append(description)
                self.text_lang.append(self.detect_lang(
                        description))
                continue

            # Check for semantic overlap between title and caption
            # to avoid redundancy.
            if self.check_semantic_overlap(
                    string1=caption,string2=title) > \
                    SEMANTIC_OVERLAP_THRESHOLD:

                self.merged_text.append(title)
                self.text_lang.append(self.detect_lang(
                        text=title))
                continue

            merged_text = self.merge_texts(
                    string1=title,
                    string2=caption
                )

            # Create an additional attribute indicating
            # the text language
            self.merged_text.append(merged_text)
            self.text_lang.append(self.detect_lang(
                text=merged_text))

        self.data[MERGED_TEXTS] = self.merged_text
        self.data[TEXT_LANGUAGE] = self.text_lang
