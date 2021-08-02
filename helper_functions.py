import sys

def null_count(df):
    return df.isna().sum().sum()

def train_test_split(df, frac):
    from sklearn.model_selection import train_test_split
    split_df = train_test_split(df, random_state = 42, train_size = frac)
    return split_df