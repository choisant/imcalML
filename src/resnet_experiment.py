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

#project specific
from imcal import *
from resnet import ResNet18
from resnet import ResNet34
from machine_learning import *

#Parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--savepath', type=str, required=True, help="Where to save .hd5 output.")
parser.add_argument('-N', type=int, default = 5, 
                        help="Number of runs of the experiment. Default=5.")

args = parser.parse_args()
N_EXPERIMENTS = args.N
#Where to save the results
SAVE_PATH = args.savepath

#variables controlled by the user. Change these to fit your specific needs.
TRAIN_N_EVENTS = 10000 #Number of events to process for each class. If higher than the available number of events an exception will be raised.
VAL_N_EVENTS = 3000
TEST_N_EVENTS = 15000 #Number of events to process for each class. If higher than the available number of events an exception will be raised.
RES = 50 #resolution

#Data specification
LABELS = ["PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL", "BH_n4_M8", "BH_n2_M10", "BH_n4_M10", "BH_n6_M10", "BH_n4_M12"]
TEST_LABELS = [f"{label}_test" for label in LABELS]
PLOT_LABELS = ["SPH_9", "BH_n4_M8", "BH_n2_M10", "BH_n4_M10", "BH_n6_M10", "BH_n4_M12"]
CLASSES = len(LABELS) #The number of output nodes in the net, equal to the number of classes
FOLDERS = ["sph", "BH", "BH", "BH", "BH", "BH"]
CUT=True
#Set data paths
if CUT:
    N_EVENTS = 10000
    TRAIN_FILENAMES = [f"{label}_res{RES}_STmin7_Nmin5_{TRAIN_N_EVENTS}_events.h5" for label in LABELS]
    VAL_FILENAMES = [f"{label}_res{RES}_STmin7_Nmin5_{VAL_N_EVENTS}_events.h5" for label in TEST_LABELS]
    TEST_FILENAMES = [f"{label}_res{RES}_STmin7_Nmin5_{TEST_N_EVENTS}_events.h5" for label in TEST_LABELS]
else:
    N_EVENTS = 10000
    TRAIN_FILENAMES = [f"{label}_res{RES}_{TRAIN_N_EVENTS}_events.h5" for label in LABELS]
    VAL_FILENAMES = [f"{label}_res{RES}_{VAL_N_EVENTS}_events.h5" for label in TEST_LABELS]
    TEST_FILENAMES = [f"{label}_res{RES}_{TEST_N_EVENTS}_events.h5" for label in TEST_LABELS]

TRAIN_DATAPATHS = [f"/disk/atlas3/data_MC/2dhistograms/{FOLDERS[i]}/{RES}/{TRAIN_FILENAMES[i]}" for i in range(CLASSES)]
TEST_DATAPATHS = [f"/disk/atlas3/data_MC/2dhistograms/{FOLDERS[i]}/{RES}/{TEST_FILENAMES[i]}" for i in range(CLASSES)]
VAL_DATAPATHS = [f"/disk/atlas3/data_MC/2dhistograms/{FOLDERS[i]}/{RES}/{VAL_FILENAMES[i]}" for i in range(CLASSES)]

#Set a unique name for the experiment
labelstring = '_'.join([str(elem) for elem in PLOT_LABELS])
if CUT:
    EXPERIMENT_NAME = f"experiment_resnet18_{str(int(time.time()))}_{labelstring}_CUT"
else: EXPERIMENT_NAME = f"experiment_resnet18_{str(int(time.time()))}_{labelstring}"
Path(f"{SAVE_PATH}/models/{EXPERIMENT_NAME}/").mkdir(parents=True, exist_ok=True)

#Set up logging
logging.basicConfig(filename=f"{SAVE_PATH}/{EXPERIMENT_NAME}.log", level=logging.INFO)
logging.info(f"Experiment started: {datetime.now()}")
logging.info(f"Running experiment with labels {LABELS}")
logging.info(f"Number of training events: {TRAIN_N_EVENTS}")
logging.info(f"Number of validation events: {VAL_N_EVENTS}")
logging.info(f"Number of testing events: {TEST_N_EVENTS}")
logging.info(f"Number of experiments to run: {N_EXPERIMENTS}")

logging.info(EXPERIMENT_NAME)

#filters
filters=["divide"]
MAX_VALUE = 200

#transforms

transforms = torch.nn.Sequential(
        torchv.transforms.RandomVerticalFlip(),
        RandomRoll(roll_axis=0)
    )
#transforms = None

#cuda
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.empty_cache()
    logging.info("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    logging.info("Running on the CPU")

#Load test data once
valpaths = [Path(path) for path in VAL_DATAPATHS]
testpaths = [Path(path) for path in TEST_DATAPATHS]

val_data = load_datasets(valpaths, DEVICE, VAL_N_EVENTS, filters, max_value=MAX_VALUE, transforms=None)
test_data = load_datasets(testpaths, DEVICE, TEST_N_EVENTS, filters, max_value=MAX_VALUE, transforms=None)

#Set up dataframe
df_labels = ["n", "resolution", "training samples", "epochs", "learning rate", "transforms", "filters", "mean accuracy", "std accuracy"]
results = pd.DataFrame(columns=df_labels)

#Function for running experiment
def experiment(df, valdata, testdata, n_train, n, epochs, lr, transforms, filters):
    scores = np.zeros(n)
    trainpaths = [Path(path) for path in TRAIN_DATAPATHS]
    traindata = load_datasets(trainpaths, DEVICE, n_train, filters, max_value=MAX_VALUE, transforms=transforms)
    for i in tqdm(range(n)):
        t0 = time.time()
        logging.info(f"Iteration {i}")
        #Set a unique name for the experiment
        if CUT:
            MODEL_NAME = f"resnet18_{str(int(time.time()))}_{labelstring}_CUT"
        else: MODEL_NAME = f"resnet18_{str(int(time.time()))}_{labelstring}"
        resnet = ResNet18(img_channels=3, num_classes=CLASSES)
        resnet.to(DEVICE)
        optimizer = optim.Adam(resnet.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=lr*10, step_size_up=5, mode="exp_range", gamma=0.85, cycle_momentum=False)
        training_results = train(resnet, traindata, valdata, 2**8, epochs, RES, DEVICE, optimizer, scheduler)
        truth, preds = predict(resnet, testdata, CLASSES, 100, RES, DEVICE)
        accuracy = accuracy_score(truth, preds, normalize=True)
        scores[i]= accuracy
        logging.info(f"Accuracy: {accuracy}")
        t1 = time.time()
        logging.info(f"Time elapsed: {int(t1-t0)} seconds: {int(t1-t0)/60} min.")
        torch.save(resnet.state_dict(), f"{SAVE_PATH}/models/{EXPERIMENT_NAME}/{MODEL_NAME}.pt")
    data = {"n":n, 
            "resolution": RES,
            "training samples": len(traindata), 
            "epochs" : epochs, 
            "learning rate" : lr, 
            "transforms"  : [transforms], 
            "filters" : filters, 
            "mean accuracy" : scores.mean(), 
            "std accuracy" : scores.std()}
    new_data = pd.DataFrame(data)
    new_df = pd.concat([df, new_data], ignore_index=True)
    return new_df, training_results, resnet

results, training_results, resnet = experiment(df=results, valdata=val_data, testdata=test_data, n_train=TRAIN_N_EVENTS,
                                                n=N_EXPERIMENTS, epochs=30, lr=0.001, transforms=transforms, filters=filters)

results_string = results.to_string()
logging.info(results_string)
logging.info(f"Experiment ended {datetime.now()}")
results.to_csv(f"{SAVE_PATH}/{EXPERIMENT_NAME}_results.csv")