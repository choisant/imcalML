import numpy as np
import awkward as ak
from fast_histogram import histogram2d
import argparse
import logging
from datetime import datetime
import sys

from imcal import *

"""
This script takes in 3-5 arguments (input root file location, where to save the new file, filename, resolution and number of events) and
outputs a .hdf5 file and a log file. The root file must be made using Delphes, and contain calorimeter and track information. The script 
can be changed to work with different root files.
The created hdf5 file has the two datasets, "image" and "labels" which can be used for machine learning purpose.
Each image has the resolution set in this script, and the label given by the filename. The functions used in the script can be found in
the imcal.py script in the same folder as this script.

Example usage:
cd src
python ./root_to_2dhists.py -f "/my/data/rootfiles/higgs.root" -s "/my/data/histograms/" -n "higgs" -r 50 -N 1000

"""

##Parser
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True, help="File to process. Must be .root")
parser.add_argument('-s', '--savepath', type=str, required=True, help="Where to save .hd5 output.")
parser.add_argument('-n', '--name', type=str, default = "empty", help="Name of file to be created (without extension). Also the label of the dataset.")
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
data_path = f"{args.file}:Delphes" 

#Set resolution
MIN_RES = 20
if args.resolution%10==0:
    RESOLUTION = args.resolution
else: 
    logging.info("Resolution must be dividable by 10.")
    logging.critical("Invalid resolution format.")
    sys.exit(1)

#Load data
clusters = load_data(data_path, MAX_EVENTS, "Tower", 
                        ["Tower.ET", "Tower.Eta", "Tower.Phi", "Tower.Eem", "Tower.Ehad", "Tower.E"])
tracks = load_data(data_path, MAX_EVENTS, "Track", 
                        ["Track.PT", "Track.Eta", "Track.Phi"])
MAX_EVENTS = len(clusters)

logging.info(f"Number of events loaded: {len(clusters)}")

#Pad Tower data
max_hits = np.max([len(event) for event in clusters["Eta"]])
logging.info(f"Padding towers to size: {max_hits}")
clusters = ak.pad_none(clusters, max_hits, axis=-1)

#Pad track data
max_hits = np.max([len(event) for event in tracks["Eta"]])
logging.info(f"Padding tracks to size: {max_hits}")
tracks = ak.pad_none(tracks, max_hits, axis=-1)

# Creating the histograms
hists_Eem = create_histograms(ak.to_numpy(clusters.Phi), ak.to_numpy(clusters.Eta), 
                                ak.to_numpy(clusters.Eem), MAX_EVENTS, RESOLUTION)
hists_Ehad = create_histograms(ak.to_numpy(clusters.Phi), ak.to_numpy(clusters.Eta), 
                                ak.to_numpy(clusters.Ehad), MAX_EVENTS, RESOLUTION)
hists_tracks = create_histograms(ak.to_numpy(tracks.Phi), ak.to_numpy(tracks.Eta), 
                                    ak.to_numpy(tracks.PT), MAX_EVENTS, RESOLUTION)

#Stack to 3 channel
images = np.stack((hists_Eem, hists_Ehad, hists_tracks), axis=-1)
logging.info(f"Image data shape: {images.shape}")

# Create meta data
meta = {
    "Resolution": RESOLUTION,
    "Events" : MAX_EVENTS,
    "Input" : data_path
}

#Save
saved = store_hists_hdf5(images, savepath, filename, meta)
logging.info(f"File saved as: {saved}")