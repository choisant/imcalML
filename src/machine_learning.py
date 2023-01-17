import torch
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torch import Tensor


#This code is heavily inspired by/copied from this tutorial; https://pythonprogramming.net/introduction-deep-learning-neural-network-pytorch/
def fwd_pass(net, X:Tensor, y:Tensor, res:int, device, optimizer, scheduler, train=False):
    if train:
        net.train()
        net.zero_grad()
    outputs = net(X.view(-1, 3, res, res).to(device))
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = F.cross_entropy(outputs, torch.argmax(y,dim=-1).to(device)) 
    if train:
        loss.backward()
        optimizer.step()
        scheduler.step()
    return acc, loss

def test(net, data, res:int, device, optimizer, scheduler, size:int = 32):
    net.eval()
    dataset = DataLoader(data, size, shuffle=True) #shuffle data and choose batch size
    X, y = next(iter(dataset)) #get a random batch
    val_acc, val_loss = fwd_pass(net, X, y, res, device, optimizer, scheduler, train=False)
    return val_acc, val_loss
    
def predict(net, testdata, num_classes, size:int, res:int, device, return_loss=False, return_values=False):
    #Returns the predictions (as class number values)
    dataset = DataLoader(testdata, size, shuffle=False) #shuffle data and choose batch size
    prediction = torch.zeros((len(dataset), size))
    truth = torch.zeros((len(dataset), size))
    losses = torch.zeros((len(dataset), size))
    values = torch.zeros((len(dataset), size, num_classes))
    i = 0
    net.eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            X, y = data
            outputs = net(X.view(-1, 3, res, res).to(device))
            values[i] = torch.softmax(outputs,dim=-1)
            losses[i] = F.cross_entropy(outputs, torch.argmax(y,dim=-1).to(device)) 
            prediction[i] = torch.argmax(outputs, dim=-1)
            truth[i] = torch.argmax(y, dim=-1)
            i = i+1

    if return_loss and not return_values:
        return torch.flatten(truth), torch.flatten(prediction), torch.flatten(losses)
    elif return_values and not return_loss:
        return torch.flatten(truth), torch.flatten(prediction), values.view(len(testdata), num_classes)
    elif return_loss and return_values:
        return torch.flatten(truth), torch.flatten(prediction), torch.flatten(losses), values.view(len(testdata), num_classes)
    else:
        return torch.flatten(truth), torch.flatten(prediction)


def shuffle_predict(net, testdata, num_classes, size:int, res:int, device, return_loss=False, return_values=False):
    #Returns the predictions (as class number values)
    dataset = DataLoader(testdata, size, shuffle=True) #shuffle data and choose batch size
    prediction = torch.zeros((len(dataset), size))
    truth = torch.zeros((len(dataset), size))
    losses = torch.zeros((len(dataset), size))
    values = torch.zeros((len(dataset), size, num_classes))
    i = 0
    net.eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            X, y = data
            outputs = net(X.view(-1, 3, res, res).to(device))
            values[i] = torch.softmax(outputs,dim=-1)
            losses[i] = F.cross_entropy(outputs, torch.argmax(y,dim=-1).to(device)) 
            prediction[i] = torch.argmax(outputs, dim=-1)
            truth[i] = torch.argmax(y, dim=-1)
            i = i+1

    if return_loss and not return_values:
        return torch.flatten(truth), torch.flatten(prediction), torch.flatten(losses)
    elif return_values and not return_loss:
        return torch.flatten(truth), torch.flatten(prediction), values.view(len(testdata), num_classes)
    elif return_loss and return_values:
        return torch.flatten(truth), torch.flatten(prediction), torch.flatten(losses), values.view(len(testdata), num_classes)
    else:
        return torch.flatten(truth), torch.flatten(prediction)

def train(net, traindata, testdata, size:int, epochs:int, res:int, device, optimizer, scheduler):
    dataset = DataLoader(traindata, size, shuffle=True)
    df_labels = ["Loss", "Accuracy", "Validation loss", "Validation accuracy", "Epoch", "Iteration"]
    df_data = [[0], [0], [0], [0], [0], [0]]
    df = pd.DataFrame(dict(zip(df_labels, df_data)))
    i = 0
    for epoch in tqdm(range(epochs)):
        for data in dataset:
            i = i+1
            X, y = data
            acc, loss = fwd_pass(net, X, y, res, device, optimizer, scheduler, train=True)
            #acc, loss = test(net, testdata, size=size)
            if i % 10 == 0:
                val_acc, val_loss = test(net, testdata, res, device, optimizer, scheduler, size)
                df_data = [float(loss), acc, float(val_loss), val_acc, epoch, i]
                new_df = pd.DataFrame(dict(zip(df_labels, df_data)), index=[0])
                df = pd.concat([df, new_df], ignore_index=True)
            
    return df