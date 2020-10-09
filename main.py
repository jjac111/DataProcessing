from utils import *


merged = merge_files()

normalized = normalize(merged)

#plot_normality(normalized)

sorted_w = relief_test(normalized)

pearson_coefs, to_compare = pearson_test(normalized)

scores = score_features(normalized, sorted_w, to_compare)