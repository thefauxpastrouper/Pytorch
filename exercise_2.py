import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn


weight = 0.3
bias = 0.9
start= 0
end = 1
step = 0.01

X = torch.arange(start,end,step)
y = X*weight+bias
print(len(X),len(y))

Train_split = int(0.8*len(X))
x_train, y_train = X[:Train_split],y[:Train_split]
x_test, y_test = X[Train_split:], y[Train_split:]

def plot_predictions(train_data = x_train,
                     train_labels = y_train,
                     test_data = x_test,
                     test_labels = y_test,
                     predictions = None):
    plt.scatter(train_data,train_labels,c = 'b', label = "Traning data")
    plt.scatter(test_data, test_labels, c = 'g', label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions,s = 25, c = 'r', label= "Predictions")

    plt.legend(prop={"size":14})
    plt.show()

plot_predictions()

class LinearRegressionModel(nn.Module):
    def __init__(self) :
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,dtype=torch.float32),requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float32),requires_grad=True)

    def forward(self, x:torch.tensor)->torch.tensor:
        return self.weights * x+ self.bias
torch.manual_seed(42)
model_2 = LinearRegressionModel()
print(model_2.state_dict())

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params = model_2.parameters(),lr=0.01)

epochs = 300



for epoch in range(epochs):

    model_2.train()
    y_preds = model_2(x_train)
    loss = loss_fn(y_preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_2.eval()
    with torch.inference_mode():
        test_preds = model_2(x_test)
        test_loss = loss_fn(y_test, test_preds)

    if epoch% 20==0:
        print(f"Epoch: {epoch}| train_loss: {loss}| testing loss:{test_loss}")

with torch.inference_mode():
    preds = model_2(x_test)
plot_predictions(predictions=preds)


from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

print(f"Saving the model: {MODEL_NAME}")
torch.save(obj = model_2.state_dict(),f=MODEL_SAVE_PATH)