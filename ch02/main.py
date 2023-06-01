import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=15)
plt.ylabel("Pizzas", fontsize=15)

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
plt.scatter(X, Y)


def predict(_x, w):
    return _x * w


def loss(_x, _y, w):
    all_predictions = predict(_x, w)
    all_errors = (all_predictions - _y) ** 2
    average_error = np.average(all_errors)
    return average_error


def train(_x, _y, iterations, learning_rate):
    w = np.average(_y) / np.average(_x)
    for i in range(iterations):
        current_loss = loss(_x, _y, w)
        print("Iteration %4d with weight %.4f => Loss: %.6f" % (i, w, current_loss))

        if loss(_x, _y, w + learning_rate) < current_loss:
            w += learning_rate
        elif loss(_x, _y, w - learning_rate) < current_loss:
            w -= learning_rate
        else:
            return w
    raise Exception("Couldn't converge within %d iterations" % iterations)


print("loop ended")

w = (train(X, Y, 500000, -0.0001))

plt.plot(X, X * w, linestyle='solid')
plt.show()
print(predict(27, w))
