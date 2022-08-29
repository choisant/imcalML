# imcal.py
from fast_histogram import histogram2d
import numpy as np
import uproot
import h5py
import matplotlib.pyplot as plt

def apply_filters(key_list, image):
    for key in key_list:
        if key=="saturate":
            image[np.nonzero(image)] = 255
            image = image.astype(int)
            print(f"Applying {key} filter.")
    return image

def cal_image_plot(ax):
    """
    Formating of calorimeter image
    """
    ax.set_ylabel(r"$\phi$ [radians]]", fontsize=16)
    ax.set_xlabel(r"$\eta$", fontsize=16)
    #ax.set_title("Calorimeter image", fontsize=20, color="black")
    ax.tick_params(which="both", direction="inout", top=True, right=True, labelsize=14, pad=15, length=4, width=2)
    ax.tick_params(which="major", length=8)
    ax.tick_params(which="major", length=6)
    ax.minorticks_on()

def create_histograms(x, y, z, max_events, res):
    max_available_events = len(x)
    if max_available_events < max_events:
        max_events = max_available_events
    Cal = [histogram2d(x[i], y[i], 
            range=[[-np.pi, np.pi], [-5, 5]], bins=res, 
            weights=z[i]) 
            for i in range(0, max_events)]
    return Cal

#Open file in with-function will close it when you exit
def load_data(rootfile:str, n_events, branch:str, keys:list):
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

def preproc_histograms(hists:list):
    max_value = np.max([np.max(item) for item in hists])
    hists = [item/max_value for item in hists]
    return hists

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

def view_data(data, cols, num_classes, spread):
    
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
