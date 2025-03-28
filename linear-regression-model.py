import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn



weight = 0.7
bias = 0.3


start = 0
end  =1
step = 0.02


X = torch.arange(start, end, step)
y = weight * X + bias

train_split = int(0.8*len(X))


X_train, y_train = X[:train_split].unsqueeze(dim=1) , y[:train_split].unsqueeze(dim=1)

X_test, y_test = X[train_split:].unsqueeze(dim=1) , y[train_split:].unsqueeze(dim=1)


device = 'cpu'

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)



def plot_predictions(train_data = X_train , 
                     train_labels = y_train , 
                     test_data = X_test , 
                     test_labels = y_test, 
                     predictions  = None):
    plt.figure(figsize=(10,7))

    plt.scatter(train_data ,
                train_labels , 
                c="b" , s=4 , 
                label="Training Data")

    plt.scatter(test_data,
                test_labels,
                c="g",
                s=4,
                label = "Testing data")

    if predictions is not None:
        plt.scatter(test_data,
                    predictions,
                    c="r",
                    s=4,
                    label="Predictions")
    

    plt.legend({"size" : 14})




plot_predictions()



# create a linear model 

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    # use nn.Linear() for creating model params
        # linear transform
    # we use prebuilt layers

        self.linear_layer = nn.Linear(in_features = 1,
                                      out_features = 1)
        
    
    def forward(self, x : torch.Tensor) -> torch.Tensor :
        return self.linear_layer(x)
    

# set the manual seed

torch.manual_seed(42)

model = LinearRegressionModel()

model, model.state_dict()
        

# set the model to use the target device


next(model.parameters()).device

model.to(device)



# train
        # loss fn
        # optimizer
        # training loop
        # testing loop

#setup loss fn

loss_fn = nn.L1Loss() # same as MAE

# setup optimizer

optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01)

# training loop


torch.manual_seed(42)


epochs = 200

for epoch in range(epochs):
    model.train()


    # forward pass

    y_pred = model(X_train)

    # calculate the loss

    loss = loss_fn(y_pred, y_train)


    # optimizer zero grad
    optimizer.zero_grad()


    #perform backpropagation

    loss.backward()

    #optimizer step
    optimizer.step()


    # testing

    model.eval()

    with torch.inference_mode():

        test_pred = model(X_test)

        test_loss = loss_fn(test_pred , y_test)


        if epoch%10 == 0 :
            print(f"Epoch : {epoch} |  Loss : {loss} | Test Loss : {test_loss}")
        


model.state_dict()

# making and evaluating predictions


model.eval()

with torch.inference_mode():
    y_preds = model(X_test)

plot_predictions(predictions=y_preds.cpu())


# saving and loading a trained model


from pathlib import Path

# create a models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

#create model save path
MODEL_NAME = "linear_regression_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#save the model state dict

print(f"Saving model to : {MODEL_SAVE_PATH}")

torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)


#load a torch model


loaded_model = LinearRegressionModel()

loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))


#put the loaded model to device

loaded_model.to(device)

# next(loaded_model.parameters().device)

loaded_model.state_dict()


# evaluate loaded model


loaded_model.eval()

with torch.inference_mode():
    loaded_model_preds = loaded_model(X_test)

y_preds  == loaded_model_preds