import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class DataText:
    """
    Class gets filepath and take every line as word then change the list to pandas DataFrame

    Params
    ------
    file: str
        path to file

    Attributes
    ----------
    file:
        (string) path to text file with its name and extension

    """
    def __init__(self, file: str):
        self.__dataframe = pd.read_csv(file, delimiter=',')

        vectorizer = CountVectorizer()
        self.__word_bag = vectorizer.fit_transform(self.x)

    @property
    def df(self) -> pd.DataFrame:
        """returns data read from file as pandas DataFrame"""
        return self.__dataframe

    @property
    def x(self) -> pd.DataFrame:
        """returns data read from file as pandas DataFrame"""
        return self.__dataframe['Text']

    @property
    def y(self) -> pd.DataFrame:
        """returns data read from file as pandas DataFrame"""
        return self.__dataframe['Class']

    @property
    def word_bag(self):
        return self.__word_bag.get_feature_names()

    @property
    def words_count(self):
        return len(self.__word_bag.get_feature_names())

    @property
    def vectors(self):
        return self.__word_bag.toarray()

    @property
    def class_count(self):
        return self.y.nunique()
