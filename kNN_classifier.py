from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris["data"]
Y = iris["target"]

X = StandardScaler().fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def most_commn(lst):
    return max(set[lst], key=lst.count)

def eucldean_distance(point, data):
    return np.sqrt(np.sum((point-data)**2, axis = 1))

class KNeighboursClassifier():
    def __init__(self, k=5, dist_metric = eucldean_distance ):
        self.dist_metric  = dist_metric
        self.k = k
    
    
    def fit_model(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        neighbours_list = []
        for x in x_test:
            distances = self.dist_metric(x, self.x_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbours_list.append(y_sorted)

        return list[map(most_commn,neighbours_list)]

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        accuracy = sum(y_pred==y_test)/len(y_test)
        return accuracy
    

accuracies = []
ks = range(1,30)
for k in ks:
    knn = KNeighboursClassifier(k=k)
    knn.fit_model(x_train, y_train)
    accuracy = knn.evaluate(x_test, y_test)
    accuracies.append(accuracy)

# Visualize accuracy vs. k
fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Performance of knn")
plt.show()