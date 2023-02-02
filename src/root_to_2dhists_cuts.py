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
the imcal.py script in the same folder as this script. Eta(phi) is along x(y)-axis of the created histogram when viewed using the 
matplotlib imshow function.

Example usage:
cd src
python ./root_to_2dhists.py -f "/my/data/rootfiles/higgs.root" -s "/my/data/histograms/" -n "higgs" -r 50 -N 1000

"""

##Parser
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True, help="File to process. Must be .root")
parser.add_argument('-s', '--savepath', type=str, required=True, help="Where to save .hd5 output.")
parser.add_argument('--ST_min', type=str, required=True, help="Minimum total energy of high pT objects")
parser.add_argument('--N_min', type=str, required=True, help="Minimum number of high pT, low eta objects")
parser.add_argument('-n', '--name', type=str, default = "empty", help="Name of file to be created (without extension). Also the label of the dataset.")
parser.add_argument('-r', '--resolution', type=int, default = 80, 
                        help="Resolution of histograms. If not specified, it will be 80. Must be an integer*10.")
parser.add_argument('-N', type=int, default = -1, 
                        help="Number of histograms to create. If not specified, it will be the same as number of events in rootfile.")

args = parser.parse_args()

#Set up logging
logging.basicConfig(filename="root_to_2dhists_cut.log", level=logging.INFO)
logging.info(f"//{datetime.now()}//")
logging.info(f"Reading file {args.file}")

#set global variables
MAX_EVENTS = args.N #images to make
savepath = args.savepath
filename = args.name
data_path = f"{args.file}:Delphes" 
ST_min = float(args.ST_min)
N_min = int(args.N_min)
min_pt = 70
max_eta = 2.4

#Set resolution
MIN_RES = 20
if args.resolution%10==0:
    RESOLUTION = args.resolution
else: 
    logging.info("Resolution must be dividable by 10.")
    logging.critical("Invalid resolution format.")
    sys.exit(1)

#Functions
def cut_pt_eta(array, min_pt, max_eta):
    array = ak.pad_none(array, 1, axis=-1)
    array = array[array.PT > min_pt]
    array = array[abs(array.Eta) < max_eta]
    n = np.array([len(event) for event in array.PT])
    return array, n

def calculate_ST(jets, muons, electrons, photons, met):
    ST = np.zeros(MAX_EVENTS)
    jet_sum = np.sum(jets.PT, axis=-1)/1000
    muon_sum = np.sum(muons.PT, axis=-1)/1000
    electron_sum = np.sum(electrons.PT, axis=-1)/1000
    photon_sum = np.sum(photons.PT, axis=-1)/1000
    met_sum = np.sum(met.MET, axis=-1)/1000
    ST = jet_sum + muon_sum + electron_sum + photon_sum + met_sum
    return ST

#Load data
clusters = load_data(data_path, MAX_EVENTS, "Tower", 
                        ["Tower.ET", "Tower.Eta", "Tower.Phi", "Tower.Eem", "Tower.Ehad", "Tower.E"])

tracks = load_data(data_path, MAX_EVENTS, "Track", 
                        ["Track.PT", "Track.Eta", "Track.Phi"])

jets = load_data(data_path, MAX_EVENTS, "Jet", 
                            ["Jet.PT", "Jet.Eta", "Jet.Phi"])
                
met = load_data(data_path, MAX_EVENTS, "MissingET", 
                        ["MissingET.MET", "MissingET.Eta", "MissingET.Phi"])

electrons = load_data(data_path, MAX_EVENTS, "Electron", 
                        ["Electron.PT", "Electron.Eta", "Electron.Phi", "Electron.Charge"])

muons = load_data(data_path, MAX_EVENTS, "Muon", 
                        ["Muon.PT", "Muon.Eta", "Muon.Phi", "Muon.Charge"])

photons = load_data(data_path, MAX_EVENTS, "Photon", 
                        ["Photon.PT", "Photon.Eta", "Photon.Phi"])

jets, n_jets = cut_pt_eta(jets, min_pt, max_eta)
electrons, n_electrons = cut_pt_eta(electrons, min_pt, max_eta)
muons, n_muons = cut_pt_eta(muons, min_pt, max_eta)
photons, n_photons = cut_pt_eta(photons, min_pt, max_eta)

ST = calculate_ST(jets, muons, electrons, photons, met)
N = np.array(n_jets) + np.array(n_electrons) + np.array(n_muons) + np.array(n_photons)
ST_idx = np.nonzero(ST > ST_min)
N_idx = np.nonzero(N > N_min)
cut_idx = np.intersect1d(ST_idx, N_idx)

#Apply cut
logging.info(f"Applying ST min cut: {ST_min} and N min cut: {N_min}")
clusters = clusters[cut_idx]
tracks = tracks[cut_idx]
CUT_EVENTS = len(clusters)

logging.info(f"Number of events loaded: {MAX_EVENTS}")
logging.info(f"Number of events after cut: {CUT_EVENTS}")

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
    "Input" : data_path,
    "ST_min" : ST_min,
    "N_min" : N_min
}

#Save
saved = store_hists_hdf5(images, savepath, filename, meta, cut=True)
logging.info(f"File saved as: {saved}")