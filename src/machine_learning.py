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
    """
    This function controls the machine learning steps, depending on if we are in training mode or not.
    """
    if train:
        net.train()
        net.zero_grad()
    #swap last axes, new config: channel, y, x
    X = torch.swapaxes(X, -3, -1)
    outputs = net(X.view(-1, 3, X.shape[-2], X.shape[-1]).to(device))
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = F.cross_entropy(outputs, torch.argmax(y,dim=-1).to(device)) 
    if train:
        loss.backward()
        optimizer.step()
        scheduler.step()
    return acc, loss

def test(net, data, res:int, device, optimizer, scheduler, size:int = 32):
    """
    Calculates the accuracy and the loss of the model for a random batch.
    """
    net.eval()
    dataset = DataLoader(data, size, shuffle=True) #shuffle data and choose batch size
    X, y = next(iter(dataset)) #get a random batch
    val_acc, val_loss = fwd_pass(net, X, y, res, device, optimizer, scheduler, train=False)
    return val_acc, val_loss
    
def predict(net, testdata, num_classes, size:int, res:int, device, return_loss=False, return_conf=False, return_fc=False):
    """
    Calculates the accuracy and the loss of the model in testing mode.
    If return_loss is True, it will return the loss for each datapoint.
    It can also return the softmax values of the raw output from the model.
    Does not shuffle the data.
    """
    dataset = DataLoader(testdata, size, shuffle=False) #shuffle data and choose batch size
    prediction = torch.zeros((len(dataset), size))
    truth = torch.zeros((len(dataset), size))
    if return_loss:
        losses = torch.zeros((len(dataset), size))
    if return_conf:
        confidences = torch.zeros((len(dataset), size, num_classes))
    if return_fc:
        embeddings = torch.zeros((len(dataset), size, num_classes))
    i = 0
    net.eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            X, y = data
            X = torch.swapaxes(X, -3, -1)
            outputs = net(X.view(-1, 3, X.shape[-2], X.shape[-1]).to(device))
            if return_fc: 
                embeddings[i] = outputs
            if return_conf:
                confidences[i] = torch.softmax(outputs,dim=-1)
            if return_loss:
                losses[i] = F.cross_entropy(outputs, torch.argmax(y,dim=-1).to(device)) 
            prediction[i] = torch.argmax(outputs, dim=-1)
            truth[i] = torch.argmax(y, dim=-1)
            i = i+1

    if return_loss:
        if return_conf:
            if return_fc:
                return torch.flatten(truth), torch.flatten(prediction), torch.flatten(losses), confidences.view(len(testdata), num_classes), embeddings.view(len(testdata), num_classes)
            else:
                return torch.flatten(truth), torch.flatten(prediction), torch.flatten(losses), confidences.view(len(testdata), num_classes)
        else:
            return torch.flatten(truth), torch.flatten(prediction), torch.flatten(losses)
    elif return_conf:
        if return_fc:
            return torch.flatten(truth), torch.flatten(prediction), confidences.view(len(testdata), num_classes), embeddings.view(len(testdata), num_classes)
        else:
            return torch.flatten(truth), torch.flatten(prediction), confidences.view(len(testdata), num_classes) 
    elif return_fc:
        return torch.flatten(truth), torch.flatten(prediction), embeddings.view(len(testdata), num_classes)
    else:
        return torch.flatten(truth), torch.flatten(prediction)


def shuffle_predict(net, testdata, num_classes, size:int, res:int, device, return_loss=False, return_conf=False):
    """
    Calculates the accuracy and the loss of the model in testing mode.
    If return_loss is True, it will return the loss for each datapoint.
    It can also return the softmax values of the raw output from the model.
    Shuffles the data.
    """
    dataset = DataLoader(testdata, size, shuffle=True) #shuffle data and choose batch size
    prediction = torch.zeros((len(dataset), size))
    truth = torch.zeros((len(dataset), size))
    losses = torch.zeros((len(dataset), size))
    confidences = torch.zeros((len(dataset), size, num_classes))
    i = 0
    net.eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            X, y = data
            X = torch.swapaxes(X, -3, -1)
            outputs = net(X.view(-1, 3, X.shape[-2], X.shape[-1]).to(device))
            confidences[i] = torch.softmax(outputs,dim=-1)
            losses[i] = F.cross_entropy(outputs, torch.argmax(y,dim=-1).to(device)) 
            prediction[i] = torch.argmax(outputs, dim=-1)
            truth[i] = torch.argmax(y,dim=-1)
            i = i+1

    if return_loss and not return_conf:
        return torch.flatten(truth), torch.flatten(prediction), torch.flatten(losses)
    elif return_conf and not return_loss:
        return torch.flatten(truth), torch.flatten(prediction), confidences.view(len(testdata), num_classes)
    elif return_loss and return_conf:
        return torch.flatten(truth), torch.flatten(prediction), torch.flatten(losses), confidences.view(len(testdata), num_classes)
    else:
        return torch.flatten(truth), torch.flatten(prediction)

def train(net, traindata, testdata, size:int, epochs:int, res:int, device, optimizer, scheduler):
    """
    Trains the model for the number of epochs specified, using the batch size specified.
    Returns a dataframe with the stats from the training.
    """
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