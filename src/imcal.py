# imcal.py
from fast_histogram import histogram2d
import numpy as np
import uproot
import h5py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

"""
Classes specific to our data
"""

class CalorimeterDataset(Dataset):
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
        shift = torch.randint(low=0, high=list(img.shape)[0], size=(1,1)).item()
        img = torch.roll(img, shift, roll_axis)
        return img


"""
Data processing
"""

def apply_filters(key_list, image, maxvalue=2000):
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

"""
Visualisation
"""

def view_data(data, cols, num_classes:int, spread):
    
    def matrix_image_plot(ax, label):
        ax.set_ylabel(r"$\phi$ [radians]]", fontsize=12)
        ax.set_xlabel(r"$\eta$", fontsize=12)
        ax.set_title(label, fontsize=14,weight="bold")
        ax.tick_params(which="both", direction="inout", top=True, right=True, labelsize=12, pad=5, length=4, width=2)
        ax.tick_params(which="major", length=8)
        ax.tick_params(which="minor", length=6)
        ax.minorticks_on()

    k = [[i]*cols for i in range(num_classes)]
    print(k)
    for i in range(len(k)):
        row = k[i]
        row = [item*(spread) for item in row]
        row = [int(item + np.random.randint(1, high = 100)) for item in row]
        k[i] = row
    print(k)
    images = [data.images[item].cpu() for item in k]
    print("Image shape: ", images[0][0].shape)
    labels = [data.img_labels[item].cpu() for item in k]
    labels = [label.tolist() for label in labels]

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

def load_hd5_histogram(path:Path, n_events:int):
    with h5py.File(path) as f:
        print (f.keys())
        data = f["images"][0:N_EVENTS]
        #create array
        arr = np.array(data)
        print(f"Loaded data with {len(arr)} entries of shape {np.shape(arr)}")
        print(f"Check max value: {np.max(arr)}")
        #Filters (normalise etc)
        arr = apply_filters(filters, arr, maxvalue=2000)
        return Tensor(arr)


def load_datasets(input_files:list, data_path:list, n_events:int, val_pct=0.1):
    """
    Loads datasets from input files located in the folder data_path. 
    Only n_events are loaded from each file. val_pct is the percentage used for validation.
    """
    val_size = int(n_events*val_pct)
    train_size = int(n_events*(1-val_pct))
    data = [load_hd5_histogram(data_path / file, n_events) for file in input_files]
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
        torchv.transforms.RandomHorizontalFlip(),
        torchv.transforms.RandomVerticalFlip(),
        RandomRoll(0)
    )

    train_dataset = CalorimeterDataset(Cal_train, labels_train, transform=transforms)
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
    #For logging
    path = (f"{savepath}/{filename}_{n_events}_events.h5")

    # Create a new HDF5 file
    file = h5py.File(f"{savepath}/{filename}_{n_events}_events.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.IEEE_F32LE, data=images
    )
    for key, value in meta.items():
        meta_set = file.create_dataset(
            key, np.shape(value), data=value
        )
    file.close()
    return path

