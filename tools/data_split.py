import pandas as pd
from sklearn.model_selection import train_test_split


class SplitData:
    """
    Class gets data (DataFrame) and split it to train and test sets

    Params
    ------
    data_to_split: pandas.DataFrame
        to be split for train and test
    test_size: float
        size of test data part - from 0.0 to 1.0 (both not included)

    Attributes
    ----------
    train_data:
        pandas.DataFrame
    test_data:
        pandas.DataFrame

    Methods
    -------
    get_test()
        return test set as pandas DataFrame
    get_train()
        return train set as pandas DataFrame
    """
    def __init__(self, data_to_split: pd.DataFrame, test_size: float):
        assert 0.0 < test_size < 1.0
        train, test = train_test_split(data_to_split, test_size=test_size)
        self.train_data = train
        self.test_data = test

    def get_test(self) -> pd.DataFrame:
        """return test set as pandas DataFrame"""
        return self.test_data

    def get_train(self) -> pd.DataFrame:
        """return train set as pandas DataFrame"""
        return self.train_data

