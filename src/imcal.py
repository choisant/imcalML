# imcal.py
#scientific libraries and plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

class HDF5LazyDataset(data.Dataset):
    #https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, labels, device, recursive, load_data, data_cache_size=3, 
                filters=None, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.labels = labels
        self.device = device
        self.filters = filters

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            
    def __getitem__(self, index):
        # get data
        x = torch.from_numpy(self.get_data("data", index))
        if self.filters != None:
            x = self.apply_filters(self.filters, x)
        if self.transform:
            x = self.transform(x)
        # get label
        value = self.get_data("label", index)
        print(value)
        #value = value.decode()
        y = self.label_maker(value, self.labels)
        y = torch.from_numpy(y)
        return (x.to(self.device), y.to(self.device))

    def __len__(self):
        return len(self.get_data_infos('data'))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds[()], file_path)
                    
                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append({'file_path': file_path, 'type': dname, 'shape': np.shape(ds[()]), 'cache_idx': idx})

    def _load_data(self, file_path, index):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            gname = f"group_{index}"
            data = h5_file[gname]["data"][()]
            # the cache index
            idx = self._add_to_cache(data, file_path)

            # find the beginning index of the hdf5 file we are looking for
            file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

            # the data info should have the same index since we loaded it in the same way
            self.data_info[file_idx + idx]['cache_idx'] = idx
            """for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds[()], file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx"""

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp, i)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]
        
    def label_maker(self, value, labels):
    #Creates labels for the classes. The first class gets value [1, 0, .., 0], the next [0, 1, ..., 0] etc
        for i, label in enumerate(labels):
            if value==label:
                idx = i
        vector = np.zeros(len(labels))
        vector[idx] = 1
        return vector

    def apply_filters(self, key_list, image, maxvalue=2000):
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
            print("You should seriously reconsider this.")

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
            print(f"Applying {key} filter.")
            
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

def view_data(data, cols, num_classes:int, spread, res):
    
    def matrix_image_plot(ax, label):
        ax.set_ylabel(r"$\phi$ [radians]]", fontsize=12)
        ax.set_xlabel(r"$\eta$", fontsize=12)
        ax.set_title(label, fontsize=14,weight="bold")
        ax.tick_params(which="both", direction="inout", top=True, right=True, labelsize=12, pad=5, length=4, width=2)
        ax.tick_params(which="major", length=8)
        ax.tick_params(which="minor", length=6)
        ax.minorticks_on()
    images = np.zeros((num_classes, cols, res, res, 3))
    labels = [[i]*cols for i in range(num_classes)]
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
    #images = [data[item][0].cpu() for item in k]
    print("Image shape: ", images[0][0].shape)
    #labels = [data[item][1].cpu() for item in k]
    #labels = [label.tolist() for label in labels]

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
    ax.tick_params(which="both", direction="inout", top=True, right=True, labelsize=14, pad=15, length=4, width=2)
    ax.tick_params(which="major", length=8)
    ax.tick_params(which="major", length=6)
    ax.minorticks_on()

"""
Histogram creation
"""
def create_histograms(x, y, z, max_events:int, res:int):
    max_available_events = len(x)
    if max_available_events < max_events:
        max_events = max_available_events
    Cal = [histogram2d(x[i], y[i], 
            range=[[-np.pi, np.pi], [-5, 5]], bins=res, 
            weights=z[i]) 
            for i in range(0, max_events)]
    return Cal

"""
Data loading
"""

def load_data(rootfile:str, n_events:int, branch:str, keys:list):
    """
    Loads the data as awkward array. Opens the file and extracts the data before closing it again.
    """
    with uproot.open(rootfile) as file:
        valid_list = [key in file.keys() for key in keys]
        if valid_list and n_events>0:
            arr = file[branch].arrays(keys, library="ak", how="zip")[0:n_events]
            return arr[branch]
        elif valid_list and n_events<0:
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


def load_datasets_old(input_files:list, data_path:list, device, n_events:int, val_pct=0.1, filters=None):
    #Loads the data files
    val_size = int(n_events*val_pct)
    train_size = int(n_events*(1-val_pct))
    data = [load_hd5_histogram(data_path / file, n_events, filters) for file in input_files]
    #Partitions off training data
    Cal_train = torch.cat([item[0:train_size] for item in data]).float().to(device)
    labels_train = label_maker(len(data), train_size).float().to(device)
    #Testing data
    Cal_test = torch.cat([item[(train_size):(train_size+val_size)] for item in data]).float().to(device)
    labels_test = label_maker(len(data), val_size).float().to(device)
    #Check everything is ok
    print(f"Data has shape {Cal_test[0].shape}. {len(labels_train)} training images and {len(labels_test)} testing images")
    print(f"There are {len(data)} classes.")
    
    transforms = torch.nn.Sequential(
        torchv.transforms.RandomVerticalFlip(),
        #torchv.transforms.RandomHorizontalFlip(),
        RandomRoll(0)
    )

    train_dataset = CalorimeterDataset(Cal_train, labels_train, transform=transforms)
    #train_dataset = CalorimeterDataset(Cal_train, labels_train)
    test_dataset = CalorimeterDataset(Cal_test, labels_test)
    
    return train_dataset, test_dataset


"""
Data storing
"""

def store_hists_hdf5(images, savepath, filename, meta):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (MAX_EVENTS, RESOLUTION, RESOLUTION, 3) to be stored
        labels       labels array, (MAX_EVENTS, 1) to be stored
    """
    n_events = meta["Events"]
    res = meta["Resolution"]
    #For logging
    path = (f"{savepath}/{filename}_{n_events}_events.h5")
    # Create a new HDF5 file
    file = h5py.File(f"{savepath}/{filename}_res{res}_{n_events}_events.h5", "w")

    # Create datasets in the file
    for i, image in enumerate(images):
        group = file.create_group(f"group_{i}")
        dataset = group.create_dataset(
            "data", np.shape(image), h5py.h5t.IEEE_F32LE, data=image
        )
        labelset = group.create_dataset(
            "label", np.shape(filename), data=filename
        )
    file.close()
    return path

