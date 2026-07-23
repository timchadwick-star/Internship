import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch import nn
from blobs_model import model, device

#generate data, 2 clusters of blobs labelled as 0 and 1
X, y = make_blobs(n_samples=1000, centers=2, cluster_std=0.5, random_state=0)

#create pandas dataframe from data
clusters = pd.DataFrame({'X1':X[:,0], 'X2':X[:,1], 'cluster':y})

#convert to tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#make train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#create loss function
loss_fn = nn.BCEWithLogitsLoss()

#create optimiser
optimiser = torch.optim.SGD(params=model.parameters(), lr=0.1)

#function to calculate accuracy, a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

#training 


torch.manual_seed(42)
epochs = 100

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

#initialise list for results
results = []

for epoch in range(epochs):
    model.train()

    #forward pass
    y_logits = model(X_train).squeeze()

    #convert logits to predictions of blob 1 or 0
    y_pred = torch.round(torch.sigmoid(y_logits))

    #calculate loss and accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train,y_pred)

    #zero gradients
    optimiser.zero_grad()

    #calculate gradients
    loss.backward()

    #update model parameters
    optimiser.step()

    if epoch%10 == 0:    
        #test model

        model.eval()
        #forward pass
        test_logits = model(X_test).squeeze()

        #convert logits to predictions of blob 1 or 0
        test_pred = torch.round(torch.sigmoid(test_logits))

        #calculate loss and accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test,test_pred)

        #store results in list
        results.append({"Epoch": epoch, "Train loss": loss.item(), "Train accuracy": acc, "Test loss": test_loss.item(), "Test accuracy": test_acc})



#convert results to pandas dataframe, save as csv and print
results_df = pd.DataFrame(results).round(3)
results_df.to_csv("results.csv", index=False)
print(results_df)

#print results as latex table
print(results_df.to_latex())
