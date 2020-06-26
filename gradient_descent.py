if  __name__ == '__main__':
    """
        fonction d'exemple
    """
    fc = lambda x, y: (3 * x ** 2) + (x * y) + (5 * y ** 2)
    partial_derivative_x = lambda x, y: (6 * x) + y
    partial_derivative_y = lambda x, y: (10 * y) + x

    x = 10
    y = -13
    learning_rate = 0.1
    print("Fc = %s" % (fc(x, y)))
    for epoch in range(20):
        x_gradient = partial_derivative_x(x, y)
        y_gradient = partial_derivative_y(x, y)

        x = x - learning_rate * x_gradient
        y = y - learning_rate * y_gradient
        print("Fc = %s" % (fc(x, y)))
    print("\nx = %s\ny = %s" % (x, y))
