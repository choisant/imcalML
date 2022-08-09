# imcal.py
from fast_histogram import histogram2d
import numpy as np
import uproot
import h5py

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
            range=[[-np.pi, np.pi], [-2.5, 2.5]], bins=res, 
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
    filters = meta["Filters"]
    
    def make_filter_code(filters):
        if filters[0] != "":
            letter_list = [x[0]+"_" for x in filters]
            code = ""
            code = code.join(letter_list)
            code = code[:-1]
        return str(code)
    filter_code = make_filter_code(filters)
    
    #For logging
    path = (f"{savepath}/{filename}_{n_events}_events_{filter_code}.h5")


    # Create a new HDF5 file
    file = h5py.File(f"{savepath}/{filename}_{n_events}_events_{filter_code}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    for key, value in meta.items():
        meta_set = file.create_dataset(
            key, np.shape(value), data=value
        )
    file.close()
    return path

def test():
    print("Import success")