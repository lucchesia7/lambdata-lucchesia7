import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class Wrangle:
    def __init__(self):
        return

    def null_count(self, df):
        # Returns null count inside all of DF
        return df.isnull().sum().sum()


class Split(Wrangle):
    def __init__(self):
        super().__init__()

    """
    Define a Train_Test_Split with
    Training Size Variable
    """

    def tts(self, df, frac):
        return train_test_split(df, random_state=42, train_size=frac)


class Randomize(Wrangle):
    def __init__(self):
        super().__init__()

    def randomize(self, df, seed):
        return df.sample(frac=1, random_state=seed)
