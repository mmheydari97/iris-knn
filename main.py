import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import matplotlib.pyplot as plt

# loading iris dataset
iris = load_iris()

# printing features and labels
print("features names: ", iris.feature_names)
print("iris types names: ", iris.target_names)

# storing features and labels
iris_X = iris.data
iris_y = iris.target

# printing number of samples
print("shape of data: ", iris_X.shape)

# spitting train set and test set
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.2, random_state=7)

# finding optimum k for our algorithm
k_range = range(1, 20)
scores = dict()
scores_list = list()

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))

# plotting the result
plt.plot(k_range, scores_list)
plt.xlabel("Value of K")
plt.ylabel("Accuracy")


# -------- finally -------- #
# getting optimum k
knn = KNeighborsClassifier(n_neighbors=np.argmax(scores_list))

# fitting the whole dataset
knn.fit(iris_X, iris_y)

# names of iris types
types = {i: iris.target_names[i] for i in range(len(iris.target_names))}

# creating sample validation set
X_val = [[7, 5, 7, 5], [4, 4, 3, 3]]
y_val = knn .predict(X_val)

for i in range(len(X_val)):
    print(types[y_val[i]])

plt.show()
