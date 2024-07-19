from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

print(f"the first 5 X features:\n{X[:5]}")
print(f"the first 5 y labels: \n{y[:5]}")

circles = pd.DataFrame({"X1":X[:,0],
                   "X2":X[:,1],
                   "label":y})
print(circles.head(10))
print(circles.label.value_counts())

# visualize
plt.scatter(x= X[:,0], y=X[:,1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(X[:5],y[:5])

X_train,x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

print(len(X_train),len(x_test), len(y_train), len(y_test))

