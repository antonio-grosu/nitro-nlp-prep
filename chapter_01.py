import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
import numpy as np

plt.ion()


what_were_covering = {1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
}


#preparing data


#linear regression formula to make a straight line  with known parameters

weight = 0.7    

bias = 0.3

start = 0
end=1

step = 0.02


# linear regression a + bX
X = torch.arange(start,end ,step).unsqueeze(dim=1)
y = weight* X + bias


#split for training and tests sets



#create a train / test split


train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split] , y[:train_split]
X_test , y_test = X[train_split:] , y[train_split:]

def plot_predictions(train_data=X_train , train_labels = y_train , test_data = X_test, test_labels = y_test , predictions = None):
    plt.figure(figsize=(10,7))


    #plot training data in blue
    # plt.scatter(train_data ,  train_labels , c="b" , s=4 , label="Training data")

    # plt.scatter(test_data , test_labels , c ="g" , s=4 , label="Testing data")

    # if predictions is not None :
    #     plt.scatter(test_data, predictions, c="r" , s=4 , label="Predictions")

    # plt.legend(prop={"size" : 14})


plot_predictions()


# build model 


#create linear regression model class


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()



        self.weights = nn.Parameter(torch.randn(1, requires_grad = True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad = True, dtype=torch.float))


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias




# create a random seed 


torch.manual_seed(42)


model_0 = LinearRegressionModel()


# print(list(model_0.parameters()))


#list named parameters


model_0.state_dict()


#making predictions w3itg torch.inference_mode()


# verify how well it predicts y_test from X_test

# make predictions with model 

with torch.inference_mode():
    y_preds = model_0(X_test)



plot_predictions(predictions=y_preds)



# train model

# one way to measure how wrong the model's predictions are we use a loss function


# loss function :  afunction to measure how wron the model s predictions are

# optimizer :  takes into account tge loss of a model and adjusts the models paramters to improve the loss function, for torch we need a a training loop and a test loop

#loss func
loss_fn = nn.L1Loss()

#optimizer

optimizer = torch.optim.SGD(params=model_0.parameters() , lr=0.01) # stochastic gradient descent

# building a training loop and a testing loop in torch

# - loop through the data
# - forward pass -> data moving through the forward() => forward propagation
# - calculate the loss
# - optimizer zero grad
# - ( back propagation) loss backward -  move backwards though the network to calc the gradients of each of the params with respect to the loss
# - ( gradient descent ) optimizer step - use the optimizer to adjust our models params to try and improve the loss

epochs = 200

# an epoch is one loop through the data  ( hyperparameters because we set it)

epoch_count = []
loss_values = []
test_loss_values =[]


#loop thr the data
for epoch in range(epochs):
    model_0.train() # sets all params that req gradients to req gradients


    # forward pass

    y_pred = model_0(X_train)

    # calc the loss

    loss = loss_fn(y_pred, y_train)


    # optimizer zero grad

    optimizer.zero_grad()

    # perform backprop 

    loss.backward()

    # step the optimizer perform gradient descent

    optimizer.step()



    model_0.eval() # turns off gradient tracking
    # turns off diff settings not needed when testing -> dropout, batch norm layers
    with torch.inference_mode():
            #turns off gradient tracking a couple of other things not needed for testing
        # forward
        test_pred = model_0(X_test)
        # calc the loss
        test_loss = loss_fn(test_pred, y_test)
    if epoch %10 ==0 :
        epoch_count.append(epoch)

        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch : {epoch} | Loss : {loss} | Test loss : {test_loss}")
        print(model_0.state_dict())



plt.plot(epoch_count , np.array(torch.tensor(loss_values).cpu().numpy()) , label = "Train loss")
plt.plot(epoch_count , np.array(torch.tensor(test_loss_values).cpu().numpy()) , label = "Test loss values")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()


with torch.inference_mode():
    y_preds_new = model_0(X_test)

# plot_predictions(predictions=y_preds_new)
# print(model_0.state_dict())




# saving a model in torch


# 3 methods for savinf and loading.

# torch.save() save in python pickle
# torch.load() load a saved torch obj

# torch.nn.Module.load_state_dict() allows to load a model s saved state dict => recomandat



from pathlib import Path

# create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)


# create model save path

MODEL_NAME = "01_pytorch_workflow_model.pth"

MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
print(MODEL_SAVE_PATH)



print(f"Saving model : {MODEL_SAVE_PATH}")
torch.save(model_0.state_dict() ,f = MODEL_SAVE_PATH )





#loading a model in a state_dict

# to load in we have to instantiate a new instance of our model class


loaded_model_0 = LinearRegressionModel()

loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))



# make some predictions with our loaded model


loaded_model_0.eval()

with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)


print(y_preds_new == loaded_model_preds) 