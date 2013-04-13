__author__ = 'jaco'

train_path = '/home/jaco/Downloads/train.csv'
test_path = '/home/jaco/Downloads/train.csv'

def load_data(path):
    with open(path) as f:
        headers = f.next().rstrip().split(',')
        for line in f:
            values = line.rstrip().split(',')
            label, features = values[0], values[1:]
            yield label, features

import numpy as np
features = np.array([f for _, f in load_data(train_path)])
labels = np.array([l for l, _ in load_data(train_path)])

# some magical code here