# neural network classification with torch


# classification is predicting wheter some thing is 
# one thing or another




# 1 make classification data and get ready

import sklearn
from sklearn.datasets import make_circles



n_samples = 1000

# create circles

X, y = make_circles(n_samples , 
                    noise=0.03,
                    random_state=42)
# print(len(X) , len(y))


# print(f"First 5 samples of X  : \n {X[:5]}")
# print(f"First 5 samples of y  : \n {y[:5]}")

import pandas as pd

circles = pd.DataFrame({"X1" : X[:,0],
                        "X2" : X[:,1],
                        "label" : y})

circles.head(10)


import matplotlib.pyplot as plt



plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)

# we are working with toy data sets

# check input and output shapes


# print(X.shape , y.shape)

# view the first example of features and labels

X_sample = X[0]
y_sample = y[0]


# print(f"Values for one sample of X : {X_sample} and the same for y : {y_sample}")
# print(f"Shapes for one sample of X : {X_sample.shape} and for y : {y_sample.shape}")


# turn data to tensors


import torch

X = torch.from_numpy(X).type(torch.float)

y = torch.from_numpy(y).type(torch.float)


# split data into traininf and test sets

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, # 0.2 means 20% data will be test
    random_state=42
)


len(X_train), len(X_test)


# buildin a model to classify our blue and red dots


# 1 setup agnostic code so our code will run on gpu if there is one
# 2 construct a model by subclassing nn.Module
# 3 define a loss func and optimizer
# 4 create a training and test loop

#_____________________1


from  torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


#_____________________2

# subclass nn.Modules
# create 2 nn.Linear layers that are capable of handling
#   the shapes of our data
# define forward method that outlines the forward pass
# instantiate an instance of our model c

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        

        # create  2 nn.Linear layers

            # takes in 2 features and outputs 5
        # self.layer_1 = nn.Linear(in_features = 2 , out_features=5)
            # takes in 5 features from 
            # prev layer and outputs a single feature same shape as y
        # self.layer_2 = nn.Linear(in_features=5 , out_features=1)

        self.two_linear_layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=5),
            nn.Linear(in_features = 5, out_features = 1)
           
        )

    def forward(self, x):
        # return self.layer_2(self.layer_1(x))
    # -> layer 1 -> layer 2 -> output
        return self.two_linear_layers(x)


# instantiate an instance of our model and sent 
# it to the device

model = CircleModel().to(device)



# let s replicate the model above using nn.Sequential


# model_0 = nn.Sequential(
#     nn.Linear(in_features=2, out_features=5),
#     nn.Linear(in_features=5, out_features=1)
# )

# model_0
model.state_dict()


# make predictions
with torch.inference_mode():
    untrained_preds = model(X_test.to(device))


# print(len(untrained_preds), untrained_preds.shape)
# print(len(X_test) , X_test.shape)
# print(untrained_preds[:10])
# print(y_test[:10])


X_test[:10] , y_test[:10]


# which loss fn to choose?

# for regression we chose MAE or MSE

# for classification
# binary cross entropy or categorical cross entropy

# the loss fn measures how wrong your models
# predictions are

# for optimizers , two of th emost common
# and useful are SGD and ADAM


# we will use torch.nn.BECWithLogitsLoss()

loss_fn = nn.BCEWithLogitsLoss() # this
# has the sigmoid activation function built-in
#nn.BCELoss does not have the sigmoid func built in


optimzer = torch.optim.SGD(
    params = model.parameters(), 
    lr = 0.1)

# let's calculate the accuracy - out of 100 examples,
# what percentahe does our model get right?


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(
        y_true,
        y_pred
    ).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc


# training the model

# we are going to need to build
# a training loop

# 1 forward pass
# 2 calc the loss
# 3 optimizer zero grad
# 4 loss backward => backpropagation
# 5 optimizer step => gradient descent

# going from raw logits -> prediciotns probabilities
# -> prediction labels

model

# our model outputs are going to be raw logits

# we can convert these logits into prediction
# probabilities by passing them some kind
# of actiovation functions ( ex. sigmoid for binary
# classification and softmax for multiclass classification)

# then we can convert our model s prediction probabilities
# to prediction labels by rounding them or taking
# the argmax()


model.eval()
with torch.inference_mode():
    y_logits = model(X_test.to(device))[:5]

y_logits


y_pred_probs = torch.sigmoid(y_logits)

y_pred_probs



# y_pred_probs >= 0.5 , y=1
# else y=0

# find the predicted labels 

y_preds=torch.round(y_pred_probs)

y_pred_labels = torch.round(torch.sigmoid(model(X_test.to(device))[:5]))

print(torch.eq(y_preds.squeeze() , y_pred_labels.squeeze()))
print(y_preds.squeeze())


y_test[:5]

# building a training and testing loop

torch.manual_seed(42)
torch.cuda.manual_seed(42)


# set no epochs

epochs = 1000

# put data to target device

X_train, y_train = X_train.to(device) , y_train.to(device)
X_test , y_test = X_test.to(device) , y_test.to(device)



for epoch in range(epochs):

    # train

    model.train()

    y_logits = model(X_train).squeeze()

    # logits -> pred probs -> pred labels
    y_pred = torch.round(torch.sigmoid(y_logits))

    # calc the loss / accuracy

    loss = loss_fn(
        y_logits,
        y_train)
    
    acc = accuracy_fn(y_true = y_train,
                      y_pred = y_pred)
    

    # optimizer zero grad

    optimzer.zero_grad()


    #loss backward -> backpropagation

    loss.backward()


    # optimizer step -> gradient descent

    optimzer.step()

    model.eval()

    with torch.inference_mode():
        # forward pass

        test_logits = model(X_test).squeeze()

        test_pred = torch.round(torch.sigmoid(test_logits))


        test_loss = loss_fn( test_logits,
                            y_test)
        
        test_acc = accuracy_fn(y_true =y_test, y_pred=test_pred)

        if epoch %10 ==0 :
            print(f"Epoch {epoch} | Loss : {loss}, Acc : {acc} | Test Loss : {test_loss} , Test Acc : {test_acc}")
    

# make predictions and evaluate the model
# from the metrics it looks like our model is not
# learning anything

# in other words, visualize


import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo (if it's not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary


# improving a model

# add more layers
# # Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)


class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2 , out_features=10)
        self.layer_2 = nn.Linear(in_features=10 , out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
    
    def forward(self , x):
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        return self.layer_3(self.layer_2(self.layer_1(x)))
    

model_1 = CircleModelV1().to(device)

model_1


# create a loss func

loss_fn = nn.BCEWithLogitsLoss()

# create an optimizer

optimizer = torch.optim.SGD(params = model_1.parameters() , lr=0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000



X_train, y_train = X_train.to(device) , y_train.to(device)
X_test, y_test = X_test.to(device) , y_test.to(device)

# wite a training loop


for epoch in range(epochs):

    model_1.train()

    y_logits = model_1(X_train).squeeze()

    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits , y_train)

    acc = accuracy_fn(y_true = y_train, 
                      y_pred = y_pred)
    
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    model_1.eval()

    with torch.inference_mode() :

        test_logits = model_1(X_test).squeeze()

        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)

        test_acc = accuracy_fn(y_true = y_test,
                               y_pred = test_pred)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} |  Loss : {loss} | Acc : {acc:.2f} | Test loss : {test_loss :.5f} | Test Acc : {test_acc:.2f}% ")


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)


#smaller problem

weight = 0.7

bias = 0.3

start =0 
end =1
step = 0.01

X_regression = torch.arange(start,end,step).unsqueeze(dim=1)

y_regression = weight*X_regression + bias


# check the data

print(len(X_regression))


#create train and test splits

train_split = int(0.8 * len(X_regression))
X_train_regression , y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

plot_predictions(train_data=X_train_regression,
                 train_labels=y_train_regression,
                 test_data=X_test_regression,
                 test_labels= y_test_regression)


# same architecture

model_2 = nn.Sequential(
    nn.Linear(in_features=1 , out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10 , out_features=1)
).to(device)

model_2



loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params = model_2.parameters(), lr=0.01)


torch.manual_seed(42)

torch.cuda.manual_seed(42)

epochs = 1000



X_train_regression, y_train_regression = X_train_regression.to(device) , y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device) , y_test_regression.to(device)


for epoch in range(epochs):
    model_2.train()

    y_pred = model_2(X_train_regression)
    loss = loss_fn(y_pred , y_train_regression)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    #tes
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn(test_pred, y_test_regression)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} |  Loss : {loss} | Test loss : {test_loss}")


model_2.eval()

with torch.inference_mode():
    y_preds = model_2(X_test_regression)

plot_predictions(
    train_data=X_train_regression.cpu(),
    train_labels=y_train_regression.cpu(),
    test_data=X_test_regression.cpu(),
    test_labels=y_test_regression.cpu(),
    predictions=y_preds.cpu()

)


#the missing piece
# what patterns could u draw with infinite amount
# of straight and non straight lines




#make and plot data


n_samples = 100

X,y = make_circles(n_samples , noise = 0.03, random_state=42)

plt.scatter(X[:,0] , X[:,1] , c = y, cmap = plt.cm.RdYlBu)


# convert data to tensors and then train 


X = torch.from_numpy(X).type(torch.float)
y=  torch.from_numpy(y).type(torch.float)

#split intro ttrain and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=0.2, 
    random_state=42 )

# build a model with non linear activation functions


class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()


        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)

        self.layer_3 = nn.Linear(in_features=10, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        #where we should put our non lienar actgivation funcs?
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


model_3 = CircleModelV2().to(device)


# setup loss and optimizer


loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model_3.parameters() , lr = 0.1)


# train model with non-linearity

#random seeds

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# put all daat on tarhet devixe

X_train, y_train = X_train.to(device) , y_train.to(device)

X_test, y_test = X_test.to(device) , y_test.to(device)


epochs = 100

for epoch in range(epochs):

    model_3.train()

    y_logits = model_3(X_train).squeeze()

    y_pred = torch.round(torch.sigmoid(y_logits))

    #logits -> pred probs -> preds

    loss = loss_fn(y_logits, y_train)

    acc = accuracy_fn(y_true = y_train, y_pred=y_pred )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_3.eval()

    with torch.inference_mode():

        test_logits = model_3(X_test).squeeze()

        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true = y_test, y_pred = test_pred)

        if epoch %100 ==0:
            print(f"Epoch {epoch} | Loss : {loss} | Acc : {acc}% | Test loss : {test_loss} |  Test acc : {test_acc}%")


model_3.eval()

with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()


y_preds[:10] , y_test[:10]

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)

plt.title("Train")
plot_decision_boundary(model_3 , X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)


#replicating non linear actiavtion funcs

A = torch.arange(-10,10,1 , dtype=torch.float32)



plt.plot(A)

def relu(x : torch.tensor) -> torch.tensor :
    return torch.maximum(torch.tensor(0) , x)

plt.plot(relu(A))


def sigmoid(x):
    return 1/ (1 + torch.exp(-x))

plt.plot(torch.sigmoid(A))