import numpy as np


##################### HELPERS ############################
def load_data(path, count):
    """
    :param count: feature count
    """
    labels, a_features, b_features = [], [], []
    with open(path) as f:
        headers = f.next().rstrip().split(',')
        for line in f:
            values = line.rstrip().split(',')
            label, a, b = values[0], values[1:count+1], values[count+1:]
            labels.append(values[0])
            a_features.append([float(x) for x in values[1:count+1]])
            b_features.append([float(x) for x in values[count+1:]])
        return np.array(labels), np.array(a_features), np.array(b_features)


################# TRAINING #######################
label, a_features, b_features = load_data(TRAIN_PATH, 11)


def predict(a_features, b_features):
    """
    Magical functions that makes all predictions.

    return: A user is more influencial than B
    rtype: bool
    """
    pass
