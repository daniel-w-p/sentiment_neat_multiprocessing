import pandas as pd


class WordsDataFrameFromTxt:
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

    Methods
    -------
    get_words()
        returns data read from file as pandas DataFrame
    """
    def __init__(self, file: str):
        self.file = file

        file_read = open(file, 'r')
        lines = file_read.read().splitlines()
        self.dataframe = pd.DataFrame(lines)
        file_read.close()

    def get_words(self) -> pd.DataFrame:
        """returns data read from file as pandas DataFrame"""
        return self.dataframe
