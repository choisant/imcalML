import uproot as ur

import awkward as ak

keys = ["Event.Number", "Tower.Eta", "Tower.Phi", "Tower.Eem", "Tower.Ehad", "Track.PT", "Track.Eta", "Track.Phi"]
path_to_root = "/disk/atlas3/data_MC/delphes/BH_n4_M10_15000events.root"
with ur.open(path_to_root) as temp_file:                

    tree = temp_file["Delphes"]            

    ar = tree.arrays(library="ak")

ar = ar[ar["Event.Number"][:, 0] == 2]
ar_list = {key: ar[key].tolist() for key in keys}

with ur.recreate("BH_n4_M10_event_id_2.root") as f:

    f["Delphes"] = ar_list