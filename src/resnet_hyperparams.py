import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss as log_loss

#scientific libraries and plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#torch specific
import torch
import torchvision as torchv
import torch.optim as optim

#other libraries
import time
import logging
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score
import argparse
from tqdm import tqdm

from timeit import default_timer as timer
from datetime import timedelta

#project specific
from imcal import *
from machine_learning import *
from resnet import ResNet18
from resnet import ResNet34

N_EXPERIMENTS = 4

#variables controlled by the user. Change these to fit your specific needs.
TRAIN_N_EVENTS = 10000 #Number of events to process for each class. If higher than the available number of events an exception will be raised.
VAL_N_EVENTS = 3000

#ML constants
EPOCHS = 100 
PATIENCE = 5
filters=[None]
MAX_VALUE = 200

#Data specification
LABELS = ["PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL", "BH_n4_M8", "BH_n2_M10", "BH_n4_M10", "BH_n6_M10", "BH_n4_M12"]
TEST_LABELS = [f"{label}_test" for label in LABELS]
PLOT_LABELS = ["SPH_9", "BH_n4_M8", "BH_n2_M10", "BH_n4_M10", "BH_n6_M10", "BH_n4_M12"]
CLASSES = len(LABELS) #The number of output nodes in the net, equal to the number of classes
FOLDERS = ["sph", "BH", "BH", "BH", "BH", "BH"]

# Save results to this path 
label_string = ""
for label in PLOT_LABELS:
    label_string += str(f"{label}_")
#Where to save the results
SAVE_PATH = f"./results/resnet_hyperparams/resnet_{label_string}CUT.csv"

transforms = torch.nn.Sequential(
            torchv.transforms.RandomVerticalFlip(),
            RandomRoll(roll_axis=0))

### Hyperparameters to vary

HYPERPARAM_DICT = {
    "res" : [50], 
    "resnet_model" : ["ResNet18"],
    "lr" : [0.0001, 0.001],
    "cycle_T" : [5, 50, 500],
    "weight_decay" : [0, 0.1],
    "batchsize" : [2**8],
    "model_n" : [i for i in range(N_EXPERIMENTS)]
}

def create_hyperparam_df(hyperparam_dict):
    # Total number of combinations of hyperparams
    n_var_hyperparams = 1
    for key in hyperparam_dict.keys():
        n_var_hyperparams = n_var_hyperparams*len(hyperparam_dict[key])
    # Create dataframe
    df = pd.DataFrame(columns=hyperparam_dict.keys())
    # With all permutations
    iter_perms_down = n_var_hyperparams
    iter_perms_up = 1
    for key in hyperparam_dict.keys():
        values = hyperparam_dict[key]
        iter_perms_down = int(iter_perms_down/len(values))
        df[key] = np.array([[x]*iter_perms_down for x in values]*iter_perms_up).flatten()
        iter_perms_up = int(iter_perms_up*len(values))

    return df

### CREATE HYPERPARAM DATAFRAME ###

gridsearch_df = create_hyperparam_df(HYPERPARAM_DICT)
gridsearch_df["ACC"] = np.zeros(len(gridsearch_df))
gridsearch_df["LogLoss"] = np.zeros(len(gridsearch_df))
for label in PLOT_LABELS:
    gridsearch_df[f"ACC_{label}"] = np.zeros(len(gridsearch_df))
gridsearch_df["Epochs"] = np.zeros(len(gridsearch_df))
gridsearch_df["Training time"] = ["time"]*len(gridsearch_df)
gridsearch_df["Time"] = np.zeros(len(gridsearch_df))
gridsearch_df.to_csv(f"{SAVE_PATH}")

### Load data only if needed.
RES_VAR = 0

### Run on GPU if possible
if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        torch.cuda.empty_cache()
        print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")

### START GRID SEARCH ###
for i in range(len(gridsearch_df)):
    print(f"Gridpoint {i} out of {len(gridsearch_df)}")
    RES = gridsearch_df["res"][i]
    base_lr = gridsearch_df["lr"][i]
    batchsize = gridsearch_df["batchsize"][i]
    weight_decay = gridsearch_df["weight_decay"][i]
    step_size = gridsearch_df["cycle_T"][i]

    ### Load data only if needed.
    if RES != RES_VAR:
        #Set data paths
        TRAIN_FILENAMES = [f"{label}_res{RES}_STmin7_Nmin5_{TRAIN_N_EVENTS}_events.h5" for label in LABELS]
        VAL_FILENAMES = [f"{label}_res{RES}_STmin7_Nmin5_{VAL_N_EVENTS}_events.h5" for label in TEST_LABELS]

        TRAIN_DATAPATHS = [f"/disk/atlas3/data_MC/2dhistograms/{FOLDERS[i]}/{RES}/{TRAIN_FILENAMES[i]}" for i in range(CLASSES)]
        VAL_DATAPATHS = [f"/disk/atlas3/data_MC/2dhistograms/{FOLDERS[i]}/{RES}/{VAL_FILENAMES[i]}" for i in range(CLASSES)]

        # Load data
        if RES == 50:
            valpaths = [Path(path) for path in VAL_DATAPATHS]
            val_data = load_datasets(valpaths, DEVICE, VAL_N_EVENTS, filters, max_value=MAX_VALUE, transforms=None)
            trainpaths = [Path(path) for path in TRAIN_DATAPATHS]
            train_data = load_datasets(trainpaths, DEVICE, TRAIN_N_EVENTS, filters, max_value=MAX_VALUE, transforms=transforms)
        else:
            #Load lazily
            valpaths = [Path(path) for path in VAL_DATAPATHS]
            trainpaths = [Path(path) for path in TRAIN_DATAPATHS]
            val_data = Hdf5Dataset(valpaths, TEST_LABELS, DEVICE, 
                                    shuffle=True, filters=filters, transform=None, event_limit=VAL_N_EVENTS)
            train_data = Hdf5Dataset(trainpaths, LABELS, DEVICE, 
                                    shuffle=True, filters=filters, transform=transforms, event_limit=TRAIN_N_EVENTS)
        RES_VAR = RES

    # Initialize model
    if gridsearch_df["resnet_model"][i] == "ResNet18":
        resnet = ResNet18(img_channels=3, num_classes=CLASSES)
        MODEL_NAME = "ResNet18"
    else:
        resnet = ResNet34(img_channels=3, num_classes=CLASSES)
        MODEL_NAME = "ResNet34"
    resnet.to(DEVICE)
    optimizer = optim.Adam(resnet.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=base_lr*10, 
                                                  step_size_up=step_size, mode="exp_range", gamma=0.85, cycle_momentum=False)
    start_train = timer()
    # Train model
    training_results = train_classifier(resnet, train_data, val_data, int(batchsize), EPOCHS, DEVICE, optimizer, scheduler, early_stopping=PATIENCE)
    end_train = timer()
    
    #Evaluate model
    truth, logits = predict_classifier(resnet, val_data, CLASSES, 100, DEVICE)
    confidences = torch.softmax(logits, dim=-1)
    df_results = pd.DataFrame(columns=["index", "Truth", "Prediction", "Confidence", "Correct"])
    df_results["index"] = np.arange(0, len(truth))
    df_results["Truth"] = truth
    df_results["Confidence"], df_results["Prediction"] = torch.max(confidences, axis=-1)
    for j in range(len(PLOT_LABELS)):
        df_results[f"{j}"] = confidences[:,j]
    correct_list = np.array([True]*len(df_results))
    correct_list[df_results["Prediction"] != df_results["Truth"]] = False
    df_results["Correct"] = correct_list

    gridsearch_df.loc[i, "ACC"] = accuracy_score(df_results["Truth"], df_results["Prediction"], normalize=True)
    gridsearch_df.loc[i, "LogLoss"] = log_loss(df_results["Truth"], df_results[[f"{j}" for j in range(len(LABELS))]],
                                               labels=[j for j in range(len(LABELS))], normalize=True)
    cf_matrix = confusion_matrix(df_results["Truth"], df_results["Prediction"], normalize="true", labels=[j for j in range(len(LABELS))])
    for j in range(len(PLOT_LABELS)):
        gridsearch_df.loc[i, f"ACC_{PLOT_LABELS[j]}"] = cf_matrix[j,j]
    gridsearch_df.loc[i, "Epochs"] = training_results["Epoch"].values[-1] + 1
    total_time = timedelta(seconds=end_train-start_train)
    gridsearch_df.loc[i, "Training time"] = str(total_time)
    gridsearch_df.loc[i, "Time"] = end_train

    ### SAVE RESULTS EVERY ITERATION ###
    gridsearch_df.to_csv(f"{SAVE_PATH}", mode='w')


