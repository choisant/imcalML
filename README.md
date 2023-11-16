# imcalML repository

Paper is out on arXiv!
[https://arxiv.org/abs/2310.15227](https://arxiv.org/abs/2310.15227)
## Requirements

Users should have a basic understanding of
* Git
* python
* jupyter notebook
* Linux servers


## Setup instructions

** Instructions are for people working on a linux machine **

## Log in and setup

```
git clone https://github.com/choisant/imcalML
```

This repository requires you to work in a virtual environment. A common tool for managing such environments for python is the [Anaconda](https://www.anaconda.com/) software. We do not need the extensive features and packages that come with the full Anaconda package, so it is recommended to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you do not already have Anaconda installed.


```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

## Virtual environments
Now we need to create our virtual environments. For the notebooks which utilise lumin we should use python 3.6 and some specific package versions. We include the jupyter notebook package as well. Always remember to activate the environment at the start of a work session, and deactivate it afterwards.

```
conda create -n imcal python=3.9 notebook
conda activate imcal
```

Our next step is to install the packages we need. When we use `pip install` inside of our environment, the packages are only installed in that environment and will not affect any other projects. Make sure everything goes as it should. You can install additional packages as required.

In this venv we can install any widely used packages we wish, as these should not conflict. Some of the packages used here are:
* numpy
* pandas
* matplotlib
* scikit
* seaborn
* fast-histograms
* pytorch
* torchvision
* tqdm
* uproot
* awkward
* h5py

## Open the notebooks

Now you are ready to go. To start working on a notebook or create your own, we need to access jupyter notebook in our browser. To do this, we use ssh-tunneling. Make sure you are in the correct virtual environment.

```
cd imcalML
jupyter notebook --no-browser
```
You will see an output like this:

```
[C 10:17:24.324 NotebookApp]

    To access the notebook, open this file in a browser:
        file:///home/agrefsru/.local/share/jupyter/runtime/nbserver-25432-open.html
    Or copy and paste one of these URLs:
        http://localhost:8889/?token=a18268c5f164e0dbe2e52aec3b0429ee20264ee6828a2802
     or http://127.0.0.1:8889/?token=a18268c5f164e0dbe2e52aec3b0429ee20264ee6828a2802
```
Notice the number *8889*. This is the port that the notebook is hosted on. If you are working on a remote server, you want to forward this to your own localhost:1234 (or any other available port number). To do this, open a new terminal window and type:


```
ssh -L 1234:localhost:8889 -y <username>@<server>
```
Remember to change out username and X with your own username and the server name. Type your password and press enter. Now you should be able to open your browser and paste in the second link from above, changing out the port number to the one you are using on your local machine.

```
http://localhost:1234/?token=a18268c5f164e0dbe2e52aec3b0429ee20264ee6828a2802
```
If you did everything correctly you now have access to all the notebooks in your browser and can start working!
## Visual Studio Code

For bigger projects, it can be more convenient to work in a code editor like VSC. Make sure you set up the virtual environments correctly. Select the appropriate environment as your "Kernel" for the notebook.

You can also connect to the jupyter notebook kernel on a remote server using the Remote-SSH Plugin in VSC. See this [tutorial](https://www.digitalocean.com/community/tutorials/how-to-use-visual-studio-code-for-remote-development-via-the-remote-ssh-plugin). This is definitely the way to go if you want to make an extensive project with several scripts as well as notebooks.

## ROOT files
The location of files is hard coded and must of course be changed for the notebooks to run. Data is not provided with this project but can be recreated.
The files we use in this project are generated using Pythia/Herwig7/Delphes. The output is a ROOT file, which has a tree-like structure. All the available fields are described [here:](https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook/RootTreeDescription).
The project also uses hdf5 files, made with the `/src/root_to_2dhists.py` script. These are datasets containing 3 channel images using the energy deposits in the tracking system and the two calorimeter systems of the detector, their label and the event ID for each event. More information on this can be found in `/src/imcal.py` The files are generated from the Delphes ROOT files. Instructions for how to generate ROOT files yourself can be found in the `data_generation` folder.

## Create histograms for image classification

Use the src/root_to_2dhists.py script to create the image/label pairs for the image classification machine learning code.

Example usage:
```
cd src
python ./root_to_2dhists.py -f "/my/data/rootfiles/higgs.root" -s "/my/data/histograms/" -n "higgs" -r 50 -N 1000
```

## License
This work is marked with CC0 1.0. To view a copy of this license, visit [this page](http://creativecommons.org/publicdomain/zero/1.0)
Even with this license, we hope that if anyone uses this repository for their own work they will credit us: [Zenodo](https://zenodo.org/records/10033266)
DOI 10.5281/zenodo.10033265

## Our research project

This project is a collaboration between the Western Norway University of Applied Sciences, The University of Bergen and the University of Warsaw. The research project has the name "Understanding the Early Universe: interplay of theory and collider experiments". The aim of the project is to study phenomena that are crucial to understanding the evolution of the Universe, but unexplained by the current Particle Physic's paradigm: the Standard Model of Particle Physics.  More information can be found on the research project [website](https://www.fuw.edu.pl/~ksakurai/grieg/index.html?lang=en). It is funded by Norwegian Financial Mechanisms 2014-2021, grant no. 2019/34/H/ST2/00707 (GRIEG). The project is operated by the Polish National Science Centre.

You can read more about the HVL team at their website [learningdarkmatter.com](https://learningdarkmatter.com/).

Any inquiries can be sent to the author of this GitHub project Aurora Grefsrud: agre@hvl.no
