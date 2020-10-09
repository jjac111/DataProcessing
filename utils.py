import pandas as pd
import numpy as np
from config import *
from matplotlib import pyplot as plt
from scipy import stats
from itertools import combinations
from math import fabs


def merge_files():
    train = pd.read_csv(train_filename, header=None)
    test = pd.read_csv(test_filename, header=None)

    return pd.concat([train, test], ignore_index=True)


def euclidean_distance(vec1, vec2):
    assert len(vec1) == len(vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    normalized_1 = [0 for i in range(len(vec1))]
    normalized_2 = [0 for i in range(len(vec2))]

    if norm1:
        normalized_1 = np.array(vec1) / norm1
    if norm2:
        normalized_2 = np.array(vec2) / norm2

    return np.dot(normalized_1, normalized_2)


def normalize(df):
    if any(df.duplicated()):
        print('There are duplicated rows in the data.')
    else:
        print('There are NO duplicated rows in the data.')

    new_max = 1
    new_min = 0

    for col in df:
        df = df.astype('float64')
        if col == 0:
            continue
        minimum = min(df[col])
        maximum = max(df[col])
        for i, value in enumerate(df[col]):
            new_value = (value - minimum) / (maximum - minimum) * (new_max - new_min) + new_min
            df.iat[i, col] = new_value

    return df


def plot_normality(df):
    bins = [i / 10 for i in range(10 + 1)]
    for col in df:
        if col == 0:
            continue

        statistic, pvalue = stats.normaltest(df[col])
        print(
            f'Normality test for feature {col}:\n\tP-value: {pvalue}\t{"Samples do not come from a normal distribution." if pvalue < alpha else "Normally distributed."}')

        plt.hist(df[col], bins)
        plt.title(f'Histogram of distribution of feature {col}')
        plt.show()


def relief_test(df):
    # parametric tests require normality of data. This is non-parametric and allows for binary classification problems
    m = 5*len(df)
    threshold = 1 / np.sqrt(alpha*m)

    w = np.array([0 for i in range(len(df[df.columns[1:]].columns))])

    for i in range(m):
        random_sample = df.sample(1)
        x = random_sample.values[0][1:]
        y = random_sample[0].values[0]

        near_hit = (0, None)
        near_miss = (0, None)
        for j, row in df[df[0] == y].iterrows():
            x_ = row[1:]
            distance = euclidean_distance(x, x_)
            # Skip for comparison with identical vectors
            if distance == 1:
                continue
            if distance > near_hit[0]:
                near_hit = (distance, np.array(x_))

        for j, row in df[df[0] != y].iterrows():
            x_ = row[1:]
            distance = euclidean_distance(x, x_)
            # Skip for comparison with identical vectors
            if distance == 1:
                continue
            if distance > near_miss[0]:
                near_miss = (distance, np.array(x_))

        w = w - np.square(x - near_hit[1]) + np.square(x - near_miss[1])
        if i % int(m/20) == 0:
            print(f'{round(i * 100 / m)}%', end=' ')

    w = w / m

    print(f'\n\nFor a threshold {threshold}:')

    sorted_w = sorted([(i+1, weight) for i, weight in enumerate(w)], key=lambda x: x[1], reverse=True)
    for i, weight in sorted_w:
        print(f'Feature {i} weight: {round(weight, 4)}' + "\t Is considered irrelevant." if weight < threshold else "")

    return sorted_w


def pearson_test(df):

    attributes = df[df.columns[1:]]

    coefficients = pd.DataFrame(columns=list(attributes.keys()), index=list(attributes.keys()))

    for feat_a in attributes:
        for feat_b in attributes:
            mean_a = np.mean(df[feat_a])
            mean_b = np.mean(df[feat_b])
            stdev_a = np.sqrt((1 / (len(df) - 1)) * sum([(x - mean_a) ** 2 for x in df[feat_a]]))
            stdev_b = np.sqrt((1 / (len(df) - 1)) * sum([(x - mean_b) ** 2 for x in df[feat_b]]))

            coef = (1 /(stdev_a * stdev_b * (len(df) - 1))) * sum([(a - mean_a)*(b - mean_b)for a, b in zip(df[feat_a], df[feat_b])])

            coefficients.at[feat_a, feat_b] = coef

    combs = combinations(coefficients.columns, 2)

    to_compare = sorted([(a, b, round(fabs(coefficients.at[a, b]), 4)) for a, b in combs], key=lambda x: x[2])

    to_compare = pd.DataFrame(to_compare, columns=['feature_a', 'feature_b', 'pearson_coef'])

    return coefficients, to_compare


def score_features(df, relief_scores, pearson_coefs):
    relief_scores = pd.DataFrame(relief_scores, columns=['feature', 'relief'])

    scores = []
    for i, row in relief_scores.iterrows():
        feature = row['feature']
        relief_score = row['relief']
        a_coefs = pearson_coefs[pearson_coefs['feature_a'] == feature]['pearson_coef']
        b_coefs = pearson_coefs[pearson_coefs['feature_b'] == feature]['pearson_coef']
        total_pearson_coefs = sum(a_coefs) + sum(b_coefs)
        score = total_pearson_coefs / relief_score

        scores.append((feature, score))

    return list(sorted(scores, key=lambda x: x[1], reverse=False))
