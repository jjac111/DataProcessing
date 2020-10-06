import pandas as pd
from config import *


def merge_files():
    train = pd.read_csv(train_filename, header=None)
    test = pd.read_csv(test_filename, header=None)

    return pd.concat([train, test], ignore_index=True)

def normalize(df):
    new_max = 1
    new_min = 0

    for col in df:
        df = df.astype('float64')
        if col == 0:
            continue
        minimum = min(df[col])
        maximum = max(df[col])
        for i, value in enumerate(df[col]):
            new_value = (value - minimum)/(maximum - minimum) * (new_max - new_min) + new_min
            df.iat[i, col] = new_value

    return df

def reduce(df):
    pass