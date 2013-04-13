__author__ = 'jaco'

def load_data(path):
    with open(path) as f:
        headers = f.next().rstrip().split(',')
        for line in f:
            values = line.rstrip().split(',')
            label, features = values[0], values[1:]
            yield label, features

import numpy as np
features = np.array([f for _, f in load_data(TRAIN_PATH)])
labels = np.array([l for l, _ in load_data(TEST_PATH)])

# some magical code here
