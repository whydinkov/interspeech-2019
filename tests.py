import numpy as np

feat_dicts = {
    '1': [1, 2, 3],
    '2': [3, 4, 5]
}

print(np.average(list(feat_dicts.values()), axis=0).tolist())
