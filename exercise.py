import torch
import matplotlib.pyplot as plt
from torch import nn

print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = X*weight +bias

print (X[:5], y[:5],len(X),len(y))

Split_ratio = int(0.8*len(y))
x_train, y_train = X[:Split_ratio],y[:Split_ratio]
x_test, y_test = X[Split_ratio:], y[Split_ratio:]

print (len(x_train),len(y_train), len(x_test), len(y_test))

def plot_predictions(train_data=x_train, 
                     train_labels=y_train, 
                     test_data=x_test, 
                     test_labels=y_test, 
                     predictions=None):
 
    plt.figure(figsize=(10, 7))

  # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
    plt.legend(prop={"size": 14});
   # plt.show()

plot_predictions();


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1,
                                        out_features=1)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)        
torch.manual_seed(42)
model_1 = LinearRegressionModel()
print(model_1, model_1.state_dict())

with torch.inference_mode():
    y_preds = model_1(x_test)

model_1.to(device)
print(next(model_1.parameters()).device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(),lr= 0.01)

torch.manual_seed(42)

epochs = 100
x_train= x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model_1.train()
    y_pred= model_1(x_train)
    loss=loss_fn(y_pred,y_train )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_1.eval()

    with torch.inference_mode():
        test_pred = model_1(x_test)
        test_loss = loss_fn(test_pred, y_test.type(torch.float))
    
    if epochs%100 ==0:
        print(f"Epoch: {epoch} |Train loss: {loss}| Test loss: {test_loss}")

# Find our model's learned parameters
from pprint import pprint # pprint = pretty print, see: https://docs.python.org/3/library/pprint.html 
print("The model learned the following values for weights and bias:")
pprint(model_1.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")


#making predictions

model_1.eval()

with torch.inference_mode():
    y_preds = model_1(x_test)

plot_predictions(predictions = y_preds.cpu())


from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True, exist_ok = True)

MODEL_NAME = "model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)

loaded_model_1 = LinearRegressionModel()
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model_1.to(device)

print(f"Loaded model: \n{loaded_model_1}")
print(f"Model on device: \n {next(loaded_model_1.parameters()).device}")

loaded_model_1.eval()

with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(x_test)

print(y_preds==loaded_model_1_preds)