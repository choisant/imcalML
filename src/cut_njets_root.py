import argparse
import uproot
import awkward as ak
from fast_histogram import histogram2d
import pickle
import numpy as np

parser = argparse.ArgumentParser(description='Make cuts in jet files based on number of truth jets and given keys.')
parser.add_argument('--njets', '-n', type=int, help='An integer for the accumulator.', default=0)
parser.add_argument('--file', '-f', type=str, help='Source root file.', required=True)
parser.add_argument('--tree', type=str, help='Source root file.', default="")
parser.add_argument('--key', '-k', type=str, default="Clusters",
                        help='The name of the fields to keep in the new file. Format: string. Example: "Clusters"')
parser.add_argument('--subkeys', type=list, default=["pt", "eta", "phi"], 
                        help='The name of the subfields to keep in the new file. Format: list of strings. Example: ["pt", "eta"]')

parser.add_argument('--resolution', '-r', type=int, default=60, 
                        help='The resolution of the created images.')

args = parser.parse_args()
n_jets = args.njets
rootfile = args.file
treekey = args.tree
key = args.key
subkeys = args.subkeys
RESOLUTION = args.resolution

print(n_jets)
print(key, subkeys)
print("Loading data")
#Load relevant data
with uproot.open(rootfile + treekey) as file:
    events = file.arrays(library="ak", how="zip")
    clusters = events[key, subkeys]
    clusters["njets"] = ak.num(events["TruthJets_R10"])
    clusters = clusters[ak.any(clusters.njets == n_jets, axis=1)]
    print("Found ", len(clusters), " events with ", n_jets, " jets.")

print("Loaded data. Creating histograms.")
def create_histograms(array):
    Cal = [histogram2d(event.phi, event.eta, range=[[-np.pi, np.pi], [-2.5, 2.5]], bins=RESOLUTION, weights=event.pt) for event in array]
    return Cal
    
print("Created histograms. Saving data")

hists = create_histograms(clusters)
#Save file as pickles
file_name = "jets_part1_" + "njets_" + str(n_jets) + "_" + str(RESOLUTION) + "x" + str(RESOLUTION) + ".pkl"
pickle.dump(hists, open("./data/histograms/" + file_name, 'wb'))