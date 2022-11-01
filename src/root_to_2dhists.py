import numpy as np
import awkward as ak
from fast_histogram import histogram2d
from sklearn.preprocessing import normalize
import argparse
import logging
from datetime import datetime
import sys

from imcal import *

##Parser
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True, help="File to process. Must be .root")
parser.add_argument('-s', '--savepath', type=str, required=True, help="Where to save .hd5 output.")
parser.add_argument('-n', '--name', type=str, default = "empty", help="Name of file to be created.")
parser.add_argument('-r', '--resolution', type=int, default = 80, 
                        help="Resolution of histograms. If not specified, it will be 80. Must be an integer*10.")
parser.add_argument('-N', type=int, default = -1, 
                        help="Number of histograms to create. If not specified, it will be the same as number of events in rootfile.")
args = parser.parse_args()

#Set up logging
logging.basicConfig(filename="root_to_2dhists.log", level=logging.INFO)
logging.info(f"//{datetime.now()}//")
logging.info(f"Reading file {args.file}")

#set global variables
MAX_EVENTS = args.N #images to make
savepath = args.savepath
filename = args.name
data_paths = [f"{args.file}:Delphes"] #Is in list format because of too lazy to change

#Set resolution
MIN_RES = 20
if args.resolution%10==0:
    RESOLUTION = args.resolution
else: 
    logging.info("Resolution must be dividable ny 10.")
    logging.critical("Invalid resolution format.")
    sys.exit(1)

#Load data
clusters = [load_data(path, MAX_EVENTS, "Tower", 
                        ["Tower.ET", "Tower.Eta", "Tower.Phi", "Tower.Eem", "Tower.Ehad", "Tower.E"])
                        for path in data_paths]
tracks = [load_data(path, MAX_EVENTS, "Track", 
                        ["Track.PT", "Track.Eta", "Track.Phi"])
                        for path in data_paths]
MAX_EVENTS = len(clusters[0])

logging.info(f"Number of events loaded: {[len(item) for item in clusters]}")

#Pad Tower data and normalise
max_hits = np.max([np.max([len(event) for event in item["Eta"]]) for item in clusters])
logging.info(f"Padding towers to size: {max_hits}")
clusters = [ak.pad_none(item, max_hits, axis=-1) for item in clusters]

#Pad track data and pad none
max_hits = np.max([np.max([len(event) for event in item["Eta"]]) for item in tracks])
logging.info(f"Padding tracks to size: {max_hits}")
tracks = [ak.pad_none(item, max_hits, axis=-1) for item in tracks]

# Creating the histograms

hists_Eem = create_histograms(ak.to_numpy(clusters[0].Phi), ak.to_numpy(clusters[0].Eta), 
                                ak.to_numpy(clusters[0].Eem), MAX_EVENTS, RESOLUTION)
hists_Ehad = create_histograms(ak.to_numpy(clusters[0].Phi), ak.to_numpy(clusters[0].Eta), 
                                ak.to_numpy(clusters[0].Ehad), MAX_EVENTS, RESOLUTION)
hists_tracks = create_histograms(ak.to_numpy(tracks[0].Phi), ak.to_numpy(tracks[0].Eta), 
                                    ak.to_numpy(tracks[0].PT), MAX_EVENTS, RESOLUTION)

#Stack to RGB
images = np.stack((hists_Eem, hists_Ehad, hists_tracks), axis=-1)
logging.info(f"Image data shape: {images.shape}")

# Create meta data
meta = {
    "Resolution": RESOLUTION,
    "Events" : MAX_EVENTS,
    "Input" : data_paths[0]
}
#Save
saved = store_hists_hdf5(images, savepath, filename, meta)
logging.info(f"File saved as: {saved}")