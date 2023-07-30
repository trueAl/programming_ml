import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

fig, plt1 = plt.subplots()

plt1.axis([0, 50, 0, 50])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=15)
plt.ylabel("Pizzas", fontsize=15)

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
plt1.plot(X, Y, 'bo')

plt2 = plt1.twinx()
plt3 = plt1.twinx()


def predict(_x, _w, _b):
    return _x * _w + _b


def loss(_x, _y, _w, _b):
    all_predictions = predict(_x, _w, _b)
    all_errors = (all_predictions - _y) ** 2
    average_error = np.average(all_errors)
    return average_error


def train(_x, _y, iterations, learning_rate):
    w_hist = np.array([])
    b_hist = np.array([])
    _w = np.average(_y) / np.average(_x)
    _b = np.min(_y)
    for i in range(iterations):
        current_loss = loss(_x, _y, _w, _b)
        print("Iteration %4d with weight %.4f and bias %.4f => Loss: %.6f" % (i, _w, _b, current_loss))
        w_hist = np.append(w_hist, _w)
        b_hist = np.append(b_hist, _b)
        if loss(_x, _y, _w + learning_rate, _b) < current_loss:
            _w += learning_rate
        elif loss(_x, _y, _w - learning_rate, _b) < current_loss:
            _w -= learning_rate
        elif loss(_x, _y, _w, _b + learning_rate) < current_loss:
            _b += learning_rate
        elif loss(_x, _y, _w, _b - learning_rate) < current_loss:
            _b -= learning_rate
        else:
            plt2.plot(range(i + 1), w_hist, linestyle='dotted', color='red')
            plt3.plot(range(i + 1), b_hist, linestyle='dotted', color='yellow')
            return _w, _b
    raise Exception("Couldn't converge within %d iterations" % iterations)


print("loop ended")

w, b = (train(X, Y, 500000, 0.01))

plt1.plot(range(50), range(50) * w + b, linestyle='dotted')
fig.tight_layout()
plt.show()

print(predict(20, w, b))
