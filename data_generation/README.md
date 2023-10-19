# Data generation

These instructions should provide enough information for the experiment to be reproduced exactly. Data is generated by using a combination of HEP-specific tools. If you want to use the system for other event types, the methods should be similar. In general, events are produced first at the parton level, then they are hadronised and finally the detector response is simulated. The output of this is a root file, containing information about the event as it would be recorded in the particle detector.

The software used in the data generation is typical HEP software and has completely different requirements and dependencies. My recommendation is to install the setup on a server running CERN software. For each step in the production and installation, open a NEW shell environment and set up as required. The `LD_LIBRARY_PATH` variable is especially sensitive to pollution and can result in all kinds of errors.

## Sphalerons

To produce sphalerons we want to use the [instantons](https://gitlab.com/apapaefs/instantons) library add-on to the [Herwig7](https://herwig.hepforge.org/) event generator software. Generating these files require you to have a working installation of Herwig7, python 3. Instructions are available on request (as of January 2023). The sphaleron Herwig7 input file used in this project is included in this folder.

## Black holes

### Parton level production
The black hole events are generated using [BlackMax](https://blackmax.hepforge.org/). It should be installed using the same method and compiler as [LHAPDF](https://lhapdf.hepforge.org/install.html). For this experiment, both tools must be installed. BlackMax produces black holes and lets them decay. It can be connected to a LHAPDF library to access non-standard PDFs, and it can also be connected to PYTHIA to perform hadronisation. We want our events to be produced in the most similar way, so we choose the option of using LHAPDF and choose the same PDF as we use in Herwig for the sphaleron production. Instructions for how to do this are found in the README file of BlackMax. BlackMax parameters are controlled easily using the `parameters.txt` file. An example file is found in this directory. Note especially these parameters: 
```
Number_of_simulations
11000
incoming_particle(1:pp_2:ppbar_3:ee+)
1
Center_of_mass_energy_of_incoming_particle
13000
...
number_of_extra_dimensions
5
...
choose_a_pdf_file(200_to_240_cteq6)Or_>10000_for_LHAPDF
21000
...
Minimum_mass(GeV)
8000
Maxmum_mass(GeV)
18000
```
Running the program produces a Les Hauches Accord file (`BlackMaxLHArecord.lhe`). This file can be used as input to Herwig. Always produce a few hundred extra events, as some may not be accepted by Herwig and will result in an error if the requested number of events exceed the available number of viable events.

In this study, the minimum mass was set to 8000, 10000 and 12000 and the number of extra dimensions to 2, 4 and 6.

### Hadronization

The events are hadronized using Herwig7. If Herwig is working for production of sphalerons it should be fine to get it working for black holes as well. The `LHA_reader.in` file can be used as a template to see how to import a .lhe file, set the right PDF and save the output hepmc file (HepMC2!).

## Detector simulation

The detector is simulated using the fast simulaton tool [Delphes](https://cp3.irmp.ucl.ac.be/projects/delphes). Delphes should be installed using a newer gcc and python 3. Installation instructions can be found [here](https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook/QuickTour). The hepmc files produced in Herwig are of the older hepmc2 format. The ATLAS card was used to simulate the ATLAS detector. Navigate to the Delphes installation folder and run the simulation:
```
./DelphesHepMC2 ./cards/delphes_card_ATLAS.tcl /path/to/storage/folder/BH_n5_M8_10000events.root  ../HerwigBuilds/BlackMax/BH_n5_M8/BH_n5_M8.hepmc
```

## Cuts
Cuts are made to suppress standard model background. The cuts are based on [this paper:](https://arxiv.org/abs/1805.06013).
We make the cut slightly more restrictive by requiring a signal object to be defined as a jet or lepton with p_T > 70 GeV and eta < 2.4. Each event is required to have 5 or more signal objects and the sum of transverse energy from all signal objects plus the missing transverse energy (ST) must be greater than or equal to 7 TeV.
The cuts are performed after detector simulation.