import numpy as np


##################### SETTINGS ###########################
train_path = '/home/jaco/Downloads/train.csv'
test_path = '/home/jaco/Downloads/train.csv'


##################### HELPERS ############################
def load_data(path, count, with_labels=False):
    """
    :param count: feature count to load for one subset
    :param with_labels: indicate whether labels are present
    :rtype: 2/3-element tuple - [labels], first_feature_array, second_feature_array
    """
    labels, a_features, b_features = [], [], []
    with open(path) as f:
        offset = 1 if with_labels else 0
        headers = f.next().rstrip().split(',')
        for line in f:
            values = line.rstrip().split(',')
            if with_labels:
                labels.append(values[0])
            a_features.append([float(x) for x in values[offset:count+offset]])
            b_features.append([float(x) for x in values[count+offset:]])
        if with_labels:
            return np.array(labels), np.array(a_features), np.array(b_features)
        else:
            return np.array(a_features), np.array(b_features)


################# TRAINING #######################
label, a_features, b_features = load_data(train_path, 11, with_labels=True)


def predict(a_features, b_features):
    """
    Magical functions that makes all predictions.

    return: A user is more influencial than B
    rtype: bool
    """
    pass


#################### RESULTS ######################
with open('submission.csv', 'w+') as f:
    for af, bf in load_data(test_path, 11):
        a_better = predict(af, bf)
        f.write(bool(a_better)) # label '1' means A is more influential than B. 0 means B is more influential than A
