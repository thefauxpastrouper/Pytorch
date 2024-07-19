import torch
from torch import nn 
import matplotlib.pyplot as plt

weight =0.7
bias =0.3
start = 0
end =1
step =0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
Y = weight*X+bias

print(X[:10].shape)
print(Y[:10].shape)

Train_split = int(0.8*len(X))

x_train,y_train = X[:Train_split],Y[:Train_split]
x_test,y_test = X[Train_split:],Y[Train_split:]

print(len(x_train),len(y_train),len(x_test),len(y_test))

def plot_predictions(train_data=x_train,
                     train_labels=y_train,
                     test_data=x_test,
                     test_labels=y_test,
                     predictions=None):
    plt.scatter(train_data,train_labels,c="b", s=4, label="Training data")
    plt.scatter(test_data,test_labels,c='r', s=4, label="Test data")
    if predictions is not None:
        plt.scatter(test_data,predictions, c='b', label="Predictions")
    plt.legend(prop={"size":14})
    plt.show()

plot_predictions();


#model for predicting data

class LinearRegressionModel(nn.Module):
    def __init__(self) :
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,dtype=torch.float32),requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float32),requires_grad=True)

    def forward(self, x:torch.tensor)->torch.tensor:
        return self.weights * x+ self.bias
    


    # Set manual seed since nn.Parameter are randomly initialzied
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel()

# Check the nn.Parameter(s) within the nn.Module subclass we created
print(list(model_0.parameters()))

# List named parameters 
print(model_0.state_dict())

with torch.inference_mode(): 
    y_preds = model_0(x_test)

print(y_preds)

plot_predictions(predictions=y_preds);

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01)


#training loop and the testing loop
torch.manual_seed(42)

epochs = 200
train_loss_values = []
test_loss_values = []
epochs_count = []

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

    model_0.eval()
    with torch.inference_mode():
        test_pred =model_0(x_test)
        test_loss = loss_fn(test_pred, y_test.type(torch.float))

        if epoch%10 == 0:
            epochs_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train_loss: {loss} | MAE test loss: {test_loss} ")

#final visualization of the predictes values of the model after training
y_test = y_test.detach().numpy()
y_pred =model_0(x_test).detach().numpy()
plt.scatter(y_test,y_pred, c='b')
plt.show()


# Plot the loss curves
plt.plot(epochs_count, train_loss_values, label="Train loss")
plt.plot(epochs_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")


model_0.eval()

# 2. Setup the inference mode context manager
with torch.inference_mode():
  # 3. Make sure the calculations are done with the model and data on the same device
  # in our case, we haven't setup device-agnostic code yet so our data and model are
  # on the CPU by default.
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = model_0(x_test)
y_preds

plot_predictions(predictions=y_preds)