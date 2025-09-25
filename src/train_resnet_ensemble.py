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

#variables controlled by the user. Change these to fit your specific needs.
N_ENSEMBLE = 20
TRAIN_N_EVENTS = 10000 #Number of events to process for each class. If higher than the available number of events an exception will be raised.
VAL_N_EVENTS = 3000
TEST_N_EVENT = 15000

#ML constants
EPOCHS = 100 
PATIENCE = 5
filters=[None]
MAX_VALUE = 200
transforms = torch.nn.Sequential(
            torchv.transforms.RandomVerticalFlip(),
            RandomRoll(roll_axis=0))
HYPERPARAM_DICT = {
    "res" : 50, 
    "resnet_model" : "ResNet18",
    "lr" : 0.001,
    "cycle_T" : 5,
    "weight_decay" : 0,
    "batchsize" : 2**8,
}

#Data specification
TRAIN_N_EVENTS = 10000 #Number of events to process for each class.
VAL_N_EVENTS = 3000 #Number of events to process for each class.
TEST_N_EVENTS = 15000 #Number of events to process for each class.
RES = HYPERPARAM_DICT["res"] #resolution
CUT=True #Should cut be applied? Chooses different files if True.

#Data specification
LABELS = ["PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL", "BH_n4_M8", "BH_n2_M10", "BH_n4_M10", "BH_n6_M10", "BH_n4_M12"]
TEST_LABELS = [f"{label}_test" for label in LABELS]
PLOT_LABELS = ["SPH_9", "BH_n4_M8", "BH_n2_M10", "BH_n4_M10", "BH_n6_M10", "BH_n4_M12"]
CLASSES = len(LABELS) #The number of output nodes in the net, equal to the number of classes
FOLDERS = ["sph", "BH", "BH", "BH", "BH", "BH"]

#Set data paths
if CUT:
    N_EVENTS = 10000
    TRAIN_FILENAMES = [f"{label}_res{RES}_STmin7_Nmin5_{N_EVENTS}_events.h5" for label in LABELS]
    TEST_FILENAMES = [f"{label}_res{RES}_STmin7_Nmin5_15000_events.h5" for label in TEST_LABELS]
    VAL_FILENAMES = [f"{label}_res{RES}_STmin7_Nmin5_3000_events.h5" for label in TEST_LABELS]
else:
    N_EVENTS = 10000
    TRAIN_FILENAMES = [f"{label}_res{RES}_{N_EVENTS}_events.h5" for label in LABELS]
    TEST_FILENAMES = [f"{label}_res{RES}_3000_events.h5" for label in TEST_LABELS]
    VAL_FILENAMES = [f"{label}_res{RES}_3000_events.h5" for label in TEST_LABELS]

TRAIN_DATAPATHS = [f"/disk/atlas3/data_MC/2dhistograms/{FOLDERS[i]}/{RES}/{TRAIN_FILENAMES[i]}" for i in range(CLASSES)]
VAL_DATAPATHS = [f"/disk/atlas3/data_MC/2dhistograms/{FOLDERS[i]}/{RES}/{VAL_FILENAMES[i]}" for i in range(CLASSES)]
TEST_DATAPATHS = [f"/disk/atlas3/data_MC/2dhistograms/{FOLDERS[i]}/{RES}/{TEST_FILENAMES[i]}" for i in range(CLASSES)]

# Save results to this path 
label_string = ""
for label in PLOT_LABELS:
    label_string += str(f"{label}_")
#Where to save the results
SAVE_PATH = f"./results/models/thesis_ensemble/"
Path(f"{SAVE_PATH}").mkdir(parents=True, exist_ok=True)

### Run on GPU if possible
if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        torch.cuda.empty_cache()
        print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")
     
#Load data
trainpaths = [Path(path) for path in TRAIN_DATAPATHS]
valpaths = [Path(path) for path in VAL_DATAPATHS]
testpaths = [Path(path) for path in TEST_DATAPATHS]
#Load directly to speed up
train_data = load_datasets(trainpaths, DEVICE, TRAIN_N_EVENTS, filters, transforms=transforms)
val_data = load_datasets(valpaths, DEVICE, VAL_N_EVENTS, filters, transforms=None)
test_data = load_datasets(testpaths, DEVICE, TEST_N_EVENTS, filters, transforms=None)

#Prepare summary table
df = pd.DataFrame(0, index=np.arange(N_ENSEMBLE), columns=["ACC", "LogLoss", "Train time", "Epochs", "Test time"])
for label in PLOT_LABELS:
    df[f"ACC_{label}"] = np.zeros(len(df))

for i in range(N_ENSEMBLE):
    # Create model
    MODEL_NAME = f"resnet18_{str(int(time.time()))}"
    resnet = ResNet18(img_channels=3, num_classes=CLASSES)
    resnet.to(DEVICE)
    #Set optimizer, learning rate scheduler and train the model
    batchsize = HYPERPARAM_DICT["batchsize"]
    max_epochs = EPOCHS
    base_lr = HYPERPARAM_DICT["lr"]
    optimizer = optim.Adam(resnet.parameters(), lr=base_lr)
    halfperiod = HYPERPARAM_DICT["cycle_T"]
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=base_lr*10, step_size_up=halfperiod, mode="exp_range", gamma=0.85, cycle_momentum=False)
    # Train model
    start_train = time.time()
    training_results = train_classifier(resnet, train_data, val_data, batchsize, max_epochs, DEVICE, optimizer, scheduler, early_stopping=PATIENCE, val_batchsize=1024)
    end_train = time.time()
    torch.save(resnet.state_dict(), f"{SAVE_PATH}/{MODEL_NAME}.pt")
    # Test model
    start_test = time.time()
    truth, logits = predict_classifier(resnet, test_data, CLASSES, 100, DEVICE)
    confidences = torch.softmax(logits, dim=-1)
    end_test = time.time()
    #Record results
    df_results = pd.DataFrame(columns=["index", "Truth", "Prediction", "Confidence", "Correct"])
    df_results["index"] = np.arange(0, len(truth))
    df_results["Truth"] = truth
    df_results["Confidence"], df_results["Prediction"] = torch.max(confidences, axis=-1)
    for j in range(len(PLOT_LABELS)):
        df_results[f"{j}"] = confidences[:,j]
    correct_list = np.array([True]*len(df_results))
    correct_list[df_results["Prediction"] != df_results["Truth"]] = False
    df_results["Correct"] = correct_list

    df.loc[i, "ACC"] = accuracy_score(df_results["Truth"], df_results["Prediction"], normalize=True)
    df.loc[i, "LogLoss"] = log_loss(df_results["Truth"], df_results[[f"{j}" for j in range(len(LABELS))]],
                                               labels=[j for j in range(len(LABELS))], normalize=True)
    cf_matrix = confusion_matrix(df_results["Truth"], df_results["Prediction"], normalize="true", labels=[j for j in range(len(LABELS))])
    for j in range(len(PLOT_LABELS)):
        df.loc[i, f"ACC_{PLOT_LABELS[j]}"] = cf_matrix[j,j]
    df.loc[i, "Epochs"] = training_results["Epoch"].values[-1] + 1
    df.loc[i, "Train time"] = end_train-start_train
    df.loc[i, "Test time"] = end_test-start_test

    ### SAVE RESULTS EVERY ITERATION ###
    df.to_csv(f"{SAVE_PATH}/results.csv", mode='w')