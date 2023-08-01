# imcal.py
#scientific libraries and plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

#other libraries
from tqdm import tqdm
import time
import random
import os
import sys
from pathlib import Path
import h5py
import uproot
from fast_histogram import histogram2d
from random import shuffle as randomshuffle
from pathlib import Path
from typing import Optional, Callable, Union
from sklearn.metrics import confusion_matrix

#torch specific
import torch
import torchvision as torchv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import Tensor
from torch.utils import data

"""
Classes specific to our data
"""

class CalorimeterDataset(data.Dataset):
    """
    Creates dataset with histogram images and labels
    """
    def __init__(self, images, labels, transform=None):
            self.img_labels = labels
            self.images = images
            self.transform = transform
            
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class Hdf5Dataset(Dataset):
    def __init__(
		self,
		filepaths: Union[Path,str],
		labels: list[str],
        device: torch.device,
		shuffle: bool = True,
		filters: Optional[list[str]] = None,
        max_value: Optional[list[str]] = 5000,
		transform: Optional[Callable] = None,
        event_limit: Optional[int] = None
        
        ):
        r"""
		Args:
			filepaths (list[Path or str]): Path to HDF5 files
			labels (list[str]): labels corresponding to the labels in the HDF5 files
			device (torch.device): the device the calculations should be run on
			shuffle (bool): Randomize order of events. Always enable for training, 
				otherwise all batches will contain a single class only. Optionally enable
				shuffling in DataLoader
			transform (Callable, optional): Function for data transforms
			event_limit (int, optional): Limit number of events in dataset. Clips the file from the start. 
		"""
        super().__init__()
		
        self.filepaths = filepaths
        self.labels = labels
        self.device = device
        self.shuffle = shuffle
        self.filters = filters
        self.max_value = max_value
        self.transform = transform
        self.event_limit = event_limit

        # Open file descriptors and get event keys
        # Store file name along with key, to keep track

        self._files = {}
        self._event_keys = []
        for full_file_path in filepaths:
            if os.path.exists(full_file_path):
                print(f"Opening file {full_file_path}.")
                fd = h5py.File(full_file_path, 'r')
                filename = full_file_path.name.__str__()
                self._files[filename] = fd
                # Limit number of events, this always chooses the first :event_limit events.
                if event_limit:
                    max_events = len(fd.keys())
                    if event_limit > max_events:
                        print(f"Number of events requested are greater than number of events in file: {max_events}.")
                        exit()
                    else: 
                        n_events = event_limit
                    print(f"Selecting {n_events} events out of {max_events}.")
                    event_keys = list(fd.keys())[:n_events]
                else:
                    max_events = len(fd.keys())
                    print(f"Selecting all {max_events} events.")
                    event_keys = list(fd.keys())
                #Make event keys a tuple cointaining the filename and the key.
                event_keys = [(filename, key) for key in event_keys]
                self._event_keys += event_keys
            else:
                print(f'No file found in {full_file_path}')

        assert len(self._files) > 0, f'No files loaded'
		
        # Shuffle keys
        if (shuffle):
            randomshuffle(self._event_keys)

    def __len__(self):
        return len(self._event_keys)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename, key = self._event_keys[idx]
        group = self._files[filename].get(key)

        #convert label from string to tensor
        label = group.get('label')[()].decode()
        label = self.label_maker(label, self.labels)

        #convert data from numpy histogram to tensor
        data = group.get('data')[()]
        data = np.float32(data)
        # GTX cards are single precision only
        data = torch.from_numpy(data)
        data = apply_filters(self.filters, data, maxvalue=self.max_value)

        #Apply transforms
        if self.transform:
            data = self.transform(data)

        return (data.to(self.device), label.to(self.device))

    def getids(self):
        event_ids = []
        for filename, key in self._event_keys:
            group = self._files[filename].get(key)
            #Get event id
            event_id = group.get("event_id")[()]
            event_ids.append(event_id.tolist())
        event_ids = np.array(event_ids)
        return(event_ids)

    def label_maker(self, value, labels):
        #Creates labels for the classes. The first class gets value [1, 0, .., 0], the next [0, 1, ..., 0] etc
        #Only works if labels match data labels
        idx = None
        for i, label in enumerate(labels):
            if value in label:
                idx = i	
        if idx==None:
            print(f"{value}, not in {labels}")
        vector = np.zeros(len(labels))
        vector[idx] = 1
        #Outputs a tensor
        return torch.from_numpy(vector)

class RandomRoll(torch.nn.Module):
    """
    Applies a random roll to the image along a given axis.
    """
    def __init__(self, roll_axis):
        """
        Args:
            roll_axis (int): Axis to roll along. 0 -> y axis rolling, 1-> x-axis rolling
        """
        super().__init__()
        assert isinstance(roll_axis, int)
        self.roll_axis = roll_axis
        if roll_axis > 2:
            print("You should seriously reconsider this transform.")

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rolled.

        Returns:
            PIL Image or Tensor: Randomly rolled image.
        """
        roll_axis = self.roll_axis
        shift = torch.randint(low=0, high=list(img.shape)[roll_axis], size=(1,1)).item()
        img = torch.roll(img, shift, roll_axis)
        return img

"""
Data processing
"""

def apply_filters(key_list, image, maxvalue=2000):
    if key_list!=None:
        for key in key_list:
            if key=="saturate":
                image[image>maxvalue] = maxvalue
            
            #normalisation should probably always be last applied filter
            elif key=="normalise":
                image = (image/maxvalue)
            
            else:
                print(f"Cannot find {key} filter.")

    return image

def label_maker(n_classes:int, n_events:int):
    #Creates labels for the classes. The first class gets value [1, 0, .., 0], the next [0, 1, ..., 0] etc
    a = torch.zeros(n_events*n_classes, n_classes, dtype=torch.int)
    for i in range(n_classes):
        for j in range(n_events):
            a[n_events*i + j][i] = 1
    return a

"""
Visualisation
"""

def view_data(data, cols, num_classes:int, labels, res,spread):
    
    def matrix_image_plot(ax, label):
        ax.set_ylabel(r"$\phi$ [radians]]", fontsize=12)
        ax.set_xlabel(r"$\eta$", fontsize=12)
        ax.set_title(label, fontsize=14,weight="bold")
        ax.tick_params(which="both", direction="inout", top=True, right=True, labelsize=12, pad=5, length=4, width=2)
        ax.tick_params(which="major", length=8)
        ax.tick_params(which="minor", length=6)
        ax.minorticks_on()
    images = np.zeros((num_classes, cols, res, res, 3))
    labels = [[labels[i]]*cols for i in range(num_classes)]
    print(labels)
    k = [[i]*cols for i in range(num_classes)]
    print(k)
    for i in range(len(k)):
        row = k[i]
        row = [item*(spread) for item in row]
        row = [int(item + np.random.randint(1, high = 100)) for item in row]
        k[i] = row
    print(k)
    for i, row in enumerate(k):
        for j, item in enumerate(row):
            images[i][j] = data[item][0].cpu()
    print("Image shape: ", images[0][0].shape)

    fig, axs = plt.subplots(nrows = num_classes, ncols = cols, figsize = (cols*6, num_classes*6))
    for i in range (len(k)):
        for j in range(cols):
            matrix_image_plot(axs[i][j], str(labels[i][j]))
            axs[i][j].imshow(images[i][j], extent=[-5, 5, -np.pi, np.pi], aspect='auto')



def cal_image_plot(ax):
    """
    Formating of calorimeter image
    """
    ax.set_ylabel(r"$\phi$ [radians]", fontsize=16)
    ax.set_xlabel(r"$\eta$", fontsize=16)
    #ax.set_title("Calorimeter image", fontsize=20, color="black")
    ax.tick_params(which="both", direction="inout", bottom=True, left=True, labelsize=14, pad=15, length=4, width=2)
    ax.tick_params(which="major", length=6)
    ax.minorticks_on()

def cal_image_plot_paper(ax):
    """
    Formating of calorimeter image
    """
    ax.set_ylabel(r"$\phi$ [radians]", fontsize=24)
    ax.set_xlabel(r"$\eta$", fontsize=24)
    #ax.set_title("Calorimeter image", fontsize=20, color="black")
    ax.tick_params(which="both", direction="out", bottom=True, left=True, labelsize=20, pad=15, length=6, width=3)
    ax.minorticks_on()

def plot_conf_matrix(confusion, accuracy, labels):
    """
    plot confusion matrix
    """
    fig, ax = plt.subplots(figsize = (8, 8))

    #Generate the confusion matrix
    cf_matrix = confusion_matrix(confusion["Truth"], confusion["Predictions"], normalize="true")
    cf_matrix = np.round(cf_matrix, 3)
    ax = sn.heatmap(cf_matrix, annot=True, cbar=False, cmap='rocket', fmt='g',annot_kws={"size": 24})

    #ax.set_title('Confusion matrix\n\n', size=24)
    ax.set_xlabel('\nPredicted Values', size=24)
    ax.set_ylabel('Actual Values ', size=24)

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(labels, size=20)
    ax.yaxis.set_ticklabels(labels, size=20)
    ax.set_title(f"Accuracy: {round(accuracy, 2)*100}%", size=26, pad=20)

    ## Display the visualization of the Confusion Matrix.
    plt.show()
"""
Histogram creation
"""
def create_histograms(phi:np.array, eta:np.array, energy:np.array, max_events:int, res:int, max_eta:int=5):

    """
    Creates histograms based on the phi, eta and energy/pt data points in the calorimeter/track system.
    It produces a 2D histogram with input 1 along the y-axis and input 2 along the x-axis.
    """
    max_available_events = len(phi)
    if max_available_events < max_events:
        print("Number of events specified by user exceeds number of available events.")
        sys.exit()
    #The arrays are masked arrays after padding, so they must be compressed.
    hist = [np.histogram2d(phi[i].compressed(), eta[i].compressed(), 
            range=[[-np.pi, np.pi], [-max_eta, max_eta]], bins=res, 
            weights=energy[i].compressed())[0] 
            for i in range(0, max_events)]
    return hist

"""
Data loading
"""

def load_data(rootfile:str, branch:str, keys:list, n_events:int=-1):
    """
    Loads the data as awkward array. Opens the file and extracts the data before closing it again.
    """
    with uproot.open(rootfile) as file:
        valid_list = [key in file.keys() for key in keys]
        if valid_list and n_events>0:
            #print(f"Loading {n_events} events from branch {branch}, fields {keys}.")
            arr = file[branch].arrays(keys, library="ak", how="zip")[0:n_events]
            return arr[branch]
        elif valid_list and n_events<0:
            #print(f"Loading all events from branch {branch}, fields {keys}.")
            arr = file[branch].arrays(keys, library="ak", how="zip")
            return arr[branch]
        else:
            print(keys[not(valid_list)], " not present in data.")


def load_hd5_histogram(path:Path, n_events:int, filters):
    with h5py.File(path) as f:
        print (f.keys())
        data = f["images"][0:n_events]
        #create array
        arr = np.array(data)
        print(f"Loaded data with {len(arr)} entries of shape {np.shape(arr)}")
        print(f"Check max value: {np.max(arr)}")
        #Filters (normalise etc)
        arr = apply_filters(filters, arr, maxvalue=2000)
        return Tensor(arr)


def load_datasets(input_files:list, device, n_events:int, filters=None, transforms=None):
    """ 
    Dataset must be in hdf5 format:
    Event1 /group
        Data /dataset
        Label  /dataset #not used for now
    Event2 /group
        Data  /dataset
        Label  /dataset
    """
    def load_hd5_histogram(path:Path, n_events:int, filters=None):
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            if len(keys) >= n_events:
                keys = keys[0:n_events]
            else:
                print("Not enough events!")
                exit()
            data = [f[key]["data"] for key in keys]
            #create array
            arr = np.array(data)
            print(f"Loaded data with {len(arr)} entries of shape {np.shape(arr)}.")
            print(f"Check max value: {np.max(arr)}.")
            #Filters (normalise etc)
            arr = apply_filters(filters, arr, maxvalue=5000)
            return Tensor(arr)

    def label_maker(n_classes:int, n_events:int):
        #Creates labels for the classes. The first class gets value [1, 0, .., 0], the next [0, 1, ..., 0] etc
        a = torch.zeros(n_events*n_classes, n_classes, dtype=torch.int)
        for i in range(n_classes):
            for j in range(n_events):
                a[n_events*i + j][i] = 1
        return a
    print(f"Loads data with transforms {transforms} and filters {filters}")
    #Loads the data files
    data = [load_hd5_histogram(file, n_events, filters) for file in input_files]
    Cal = torch.cat([item[0:n_events] for item in data]).float().to(device)
    labels = label_maker(len(data), n_events).float().to(device)
    
    #Check everything is ok
    print(f"Data has shape {Cal[0].shape}")
    print(f"There are {len(data)} classes.")

    dataset = CalorimeterDataset(Cal, labels, transform=transforms)
    
    return dataset


"""
Data storing
"""

def store_hists_hdf5(images, savepath, filename, meta, cut=False):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (MAX_EVENTS, RESOLUTION, RESOLUTION, 3) to be stored
        labels       labels array, (MAX_EVENTS, 1) to be stored
    """
    n_events = meta["Events"]
    res = meta["Resolution"]
    eventid = meta["Event_ID"]
    
    if cut:
        ST_cut = int(meta["ST_min"])
        N_cut = meta["N_min"]

    # Create a new HDF5 file
    if cut:
        path = Path(f"{savepath}/{filename}_res{res}_STmin{ST_cut}_Nmin{N_cut}_{n_events}_events.h5")
        print(f"Removing {path}")
        try:
            os.remove(path)
        except OSError:
            pass
        file = h5py.File(path, "w")
    else:
        path = Path(f"{savepath}/{filename}_res{res}_{n_events}_events.h5")
        print(f"Removing {path}")
        try:
            os.remove(path)
        except OSError:
            pass
        file = h5py.File(path, "w")

    # Create datasets in the file
    for i, image in enumerate(images):
        group = file.create_group(f"group_{i}")
        dataset = group.create_dataset(
            "data", np.shape(image), data=image
        )
        labelset = group.create_dataset(
            "label", np.shape(filename), data=filename
        )
        eventidset = group.create_dataset(
            "event_id", np.shape(eventid[i]), data=eventid[i]
        )
    file.close()
    print(f"File saved as {path}")
    return path

