import numpy as np

def init_variables():
    """
        init model variables
    """
    weight = np.random.normal(size=2)
    bias = 0
    return weight, bias

def get_dataset():
    """
        genere the dataset
    """
    row_per_class = 5
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    healty = np.random.randn(row_per_class, 2) + np.array([2, 2])
    features = np.vstack([sick, healty])
    targets = np.concatenate((np.zeros(row_per_class), np.full(row_per_class, 1)))

    return features, targets

def pre_activation(features, weight, bias):
    """
        Compute pre activation
    """
    return np.dot(features, weight) + bias

def activation(z):
    """
        Compute activation 
    """
    return 1 / (1 + np.exp(-z))

if __name__ == '__main__':
    features, targets = get_dataset()
    weight, bias = init_variables()

    z = pre_activation(features, weight, bias)
    a = activation(z)
    for i, j in enumerate(targets):
        print("{index} : {target}   |   {a}".format(index=i, target=targets[i], a=a[i]))