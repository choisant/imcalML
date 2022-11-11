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
		path: Union[Path,str],
		labels: list[str],
        device: torch.device,
		shuffle: bool = True,
		transform: Optional[Callable] = None,
		event_limit: Optional[int] = None

	):
		r"""
		Args:
			path (Path or str): Path to directory containing HDF5 files, with .h5 suffix 
			shuffle (bool): Randomize order of events. Always enable for training, 
				otherwise all batches will contain a single class only. Optionally enable
				shuffling in DataLoader
			transform (Callable, optional): Function for data transforms
			event_limit (int, optional): Limit number of events in dataset 
		"""
		super().__init__()
		
		self.path = path
		self.labels = labels
		self.device = device
		self.shuffle = shuffle
		self.transform = transform
		self.event_limit = event_limit

		# Get file list
		if not isinstance(path, Path):
			path = Path(path)
		
		filenames = path.glob('*.h5')
		
		# Open file descriptors and get event keys
		# Store file name along with key, to keep track
		self._files = {}
		self._event_keys = []
		for full_file_path in filenames:
			fd = h5py.File(full_file_path, 'r')
			filename = full_file_path.name.__str__()
			self._files[filename] = fd
			# Limit number of events
			if event_limit:
				event_keys = list(fd.keys())[:event_limit]
			else:
				event_keys = list(fd.keys())
			event_keys = [(filename, key) for key in event_keys]
			self._event_keys += event_keys
		
		assert len(self._files) > 0, f'No files found in {path}'
		
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
		
		#data
		data = group.get('data')[()]
		# GTX cards are single precision only
		#data = data.astype(np.float32)
		data = torch.from_numpy(data)
		if self.transform:
			data = self.transform(data)

		return (data.to(self.device), label.to(self.device))

	def label_maker(self, value, labels):
		#Creates labels for the classes. The first class gets value [1, 0, .., 0], the next [0, 1, ..., 0] etc
		#Outputs a tensor
		for i, label in enumerate(labels):
			if value in label:
				idx = i
		vector = np.zeros(len(labels))
		vector[idx] = 1
		return torch.from_numpy(vector)

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
    ax.tick_params(which="both", direction="inout", top=True, right=True, labelsize=14, pad=15, length=4, width=2)
    ax.tick_params(which="major", length=8)
    ax.tick_params(which="major", length=6)
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
    def load_hd5_histogram(path:Path, n_events:int, filters):
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            keys = keys[0:n_events]
            data = [f[key]["data"] for key in keys]
            #create array
            arr = np.array(data)
            print(f"Loaded data with {len(arr)} entries of shape {np.shape(arr)}")
            print(f"Check max value: {np.max(arr)}")
            #Filters (normalise etc)
            arr = apply_filters(filters, arr, maxvalue=2000)
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

