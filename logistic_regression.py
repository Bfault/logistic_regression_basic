import numpy as np
import matplotlib.pyplot as plt

def init_variables():
    """
    """
    weight = np.random.normal(size=2)
    bias = 0
    return weight, bias

def get_dataset():
    """
    """
    row_per_class = 100
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    healty = np.random.randn(row_per_class, 2) + np.array([2, 2])
    features = np.vstack([sick, healty])
    targets = np.concatenate((np.zeros(row_per_class), np.full(row_per_class, 1)))

    return features, targets

def pre_activation(features, weight, bias):
    """
    """
    return np.dot(features, weight) + bias

def activation(z):
    """
    """
    return 1 / (1 + np.exp(-z))

def derivative_activation(z):
    """
    """
    return activation(z) * (1 - activation(z))

def predict(features, weight, bias):
    """
    """
    z = pre_activation(features, weight, bias)
    y = activation(z)

    return np.round(y)

def cost(predictions, targets):
    """
    """
    return np.mean((predictions - targets) ** 2)

def training(features, targets, weight, bias):
    """
    """
    epochs = int(input("Number of turns ? "))
    learning_rate = float(input(("Learning rate ? ")))

    predictions = predict(features, weight, bias)
    print("Base Accuracy", np.mean(predictions == targets))
    #plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    #plt.show()
    for epoch in range(epochs):
        if epoch % (int(epochs / 10)) == 0:
            predictions = activation(pre_activation(features, weight, bias))
            print("Cost = %s" % cost(predictions, targets))
        weight_gradient = np.zeros(weight.shape)
        bias_gradient = 0

        for feature, target in zip(features, targets):
            z = pre_activation(feature, weight, bias)
            y = activation(z)
            weight_gradient += (y - target) * derivative_activation(z) * feature
            bias_gradient += (y - target) * derivative_activation(z)
        weight = weight - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient
    predictions = predict(features, weight, bias)
    print("Training Accuracy", np.mean(predictions == targets))
    

if __name__ == '__main__':
    features, targets = get_dataset()
    weight, bias = init_variables()

    z = pre_activation(features, weight, bias)
    a = activation(z)
    training(features, targets, weight, bias)