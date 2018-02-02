from sklearn.datasets import fetch_mldata
import numpy as np

#fetch data
iris = fetch_mldata('iris')

X,y = iris["data"], iris["target"]
train_size = 100

X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

shuffle_index = np.random.permutation(train_size)

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_1 = (y_train == 1)
y_test_1 = y_test ==100

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train_1)

from sklearn.model_selection import cross_val_score