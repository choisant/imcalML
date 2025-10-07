#Torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor

import torchbnn as bnn

from tqdm import tqdm
import pandas as pd
import numpy as np
import sys

def label_maker(values, num_classes):
    labels = np.zeros((len(values), num_classes))
    for i, value in enumerate(values):
        labels[i][value] = 1
    return torch.Tensor(labels).to(torch.int)

def fwd_pass_classifier(net, X:Tensor, y:Tensor, device, optimizer, scheduler, train:bool=False, biased_class:int=-1, bias_weight:float=0.1):
    """
    This function controls the machine learning steps, depending on if we are in training mode or not.
    biased_class = -1 gives clean cross entropy without conflictual loss term
    """
    if train:
        net.train()
        net.zero_grad()
    #swap last axes, new config: channel, y, x
    X = torch.swapaxes(X, -3, -1)
    outputs = net(X.view(-1, 3, X.shape[-2], X.shape[-1]).to(device))
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    #Should definitely make this stricter
    if biased_class == -1:
        loss = F.cross_entropy(outputs, torch.argmax(y,dim=-1).to(device)) 
    else:
        weight = bias_weight
        ce_loss = F.cross_entropy(outputs, torch.argmax(y,dim=-1).to(device))
        biased_class_labels = label_maker([int(biased_class)]*len(outputs), 2).to(device)
        #torch.Tensor([biased_class]*len(outputs)).to(torch.int).to(device)
        bias_term = weight*F.cross_entropy(outputs, biased_class_labels.to(torch.float32)).to(device)
        loss = ce_loss + bias_term
    if train:
        loss.backward()
        optimizer.step()
        scheduler.step()
    return acc, loss

def train_classifier(net, traindata, testdata, batchsize:int, epochs:int, device, optimizer, scheduler, 
                     early_stopping:int=-1, val_batchsize:int=-1, biased_class:int=-1, bias_weight:float=0.1, 
                     mcd:bool=False):
    """
    Trains the model for the number of epochs specified, using the batch size specified.
    Returns a dataframe with the stats from the training.
    """
    dataset = DataLoader(traindata, batchsize, shuffle=True)
    df_labels = ["Loss", "Accuracy", "Validation loss", "Validation accuracy", "Epoch", "Iteration"]
    df_created = False
    i = 0
    patience = early_stopping #How many epochs to keep training if no improvement in validation loss
    min_loss = None
    # Check loss around 10 times each epoch
    val_loss_int = int(len(dataset)/10)
    if val_batchsize==-1:
        val_batchsize = batchsize
    for epoch in range(epochs):
        # Iterate over batches
        for data in dataset:
            i = i+1
            X, y = data
            #print(X[0], y[0])
            acc, loss = fwd_pass_classifier(net, X, y, device, optimizer, scheduler, train=True, 
                                            biased_class=biased_class, bias_weight=bias_weight)
            #acc, loss = test(net, testdata, size=size)
            if i%val_loss_int==0:
                val_acc, val_loss = test_classifier(net, testdata, device, optimizer, scheduler, val_batchsize, 
                                                    biased_class=biased_class, bias_weight=bias_weight, mcd=mcd)
                df_data = [float(loss), float(acc), float(val_loss), float(val_acc), epoch, i]
                if df_created == False:
                    df = pd.DataFrame(dict(zip(df_labels, df_data)), index=[0])
                    df_created = True
                else:
                    new_df = pd.DataFrame(dict(zip(df_labels, df_data)), index=[0])
                    df = pd.concat([df, new_df], ignore_index=True)
        #Check every epoch if we should stop
        if ((early_stopping > 0) and df_created): #If small data, we might not have validation loss yet
            if min_loss == None:
                min_loss = float(val_loss)
            elif min_loss <= df["Validation loss"].min():
                patience = patience - 1
            elif min_loss > df["Validation loss"].min():
                min_loss = df["Validation loss"].min()
                patience = early_stopping # Restart early_stopping
            if patience == 0:
                print(f"Stopping training early at epoch {epoch}")
                df.drop([0])
                return df
    df.drop([0])
    return df

def test_classifier(net, data, device, optimizer, scheduler, size:int = 32, biased_class:int=-1, 
                    bias_weight:float=0.1, mcd:bool=False):
    """
    Calculates the average accuracy and the loss of the model for the validation set, averaging over batches.
    """
    net.eval()
    if mcd:
        enable_dropout(net)
    dataset = DataLoader(data, size, shuffle=True) #shuffle data and choose batch size
    loss_list = torch.zeros(len(dataset))
    acc_list = torch.zeros(len(dataset))
    #X, y = next(iter(dataset)) #get a random batch
    with torch.no_grad():
        for i, data in enumerate(dataset):
            X, y = data
            acc_list[i], loss_list[i]  = fwd_pass_classifier(net, X, y, device, optimizer, scheduler, train=False, 
                                            biased_class=biased_class, bias_weight=bias_weight)
    val_acc = acc_list.mean()
    val_loss = loss_list.mean()
    return val_acc, val_loss
    
def predict_classifier(net, testdata, num_classes:int, size:int, device):
    """
    Calculates the accuracy and the loss of the model in testing mode.
    If return_loss is True, it will return the loss for each datapoint.
    It can also return the softmax values of the raw output from the model.
    Does not shuffle the data.
    """
    assert len(testdata)%size==0, "Please choose batch size so that testdata%size==0."

    dataset = DataLoader(testdata, size, shuffle=False) #shuffle data and choose batch size
    logits = torch.zeros((len(dataset), size, num_classes))
    truth = torch.zeros((len(dataset), size))
    i = 0
    net.eval()
    with torch.no_grad():
        for data in dataset:
            X, y = data
            X = torch.swapaxes(X, -3, -1)
            logits[i] = net(X.view(-1, 3, X.shape[-2], X.shape[-1]).to(device))
            truth[i] = torch.argmax(y, dim=-1).to(torch.int)
            i = i+1
    return torch.flatten(truth), logits.view(-1, num_classes)

def enable_dropout(net):
    """ Function to enable the dropout layers during test-time """
    for m in net.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

###### PAPER CODE ######
            
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