# imcalML repository

## Requirements

Users should have a basic understanding of
* Git
* python
* jupyter notebook
* Linux servers


## Setup instructions

** Instructions are specifically for HVL/UiB team members working on the atlasX.itf.uib.no linux server **

You need to have a CERN and ATLAS account as well as acess to the group's server to follow these instructions. The setup can also be performed on your personal computer. In that case, just skip the log-in and ssh steps.

## Log in and setup
Log in to the atlasX.itf.uib.no server through ssh (install ssh if you do not have it). Change username to your CERN username and X to the desired server (1,2,3).

```
ssh -y <username>@atlas<X>.ift.uib.no
```
You will have to enter your CERN password at this point. Now you are in your home directory. Clone this project using git (install git if you do not have it).

```
git clone https://github.com/choisant/imcalML
```

This repository requires you to work in a virtual environment. A common tool for managing such environments for python is the [Anaconda](https://www.anaconda.com/) software. We do not need the extensive features and packages that come with the full Anaconda package, so it is recommended to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you do not already have Anaconda installed.


```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Now we need to create our virtual environment which should use python 3.6. We will call the environment imcal. We include the jupyter notebook package as well. Always remember to activate the environment at the start of a work session, and deactivate it afterwards.

```
conda create -n imcal python=3.6 notebook
conda activate imcal
```
Our next step is to install the packages we need. Most of them come as required packages with the [lumin](https://github.com/GilesStrong/lumin) package, which is the base of the project. When we use `pip install` inside of our environment, the packages are only installed in that environment and will not affect any other projects.

```
pip install lumin
```
Make sure everything goes as it should. You can install additional packages as required, but note that you can break lumin if you upgrade any of its dependencies.

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
Notice the number *8889*. This is the port that the notebook is hosted on. We want to forward this to our own localhost:1234 (or any other available port number). To do this, open a new terminal window and type:


```
ssh -L 1234:localhost:8889 -y <username>@atlas<X>.ift.uib.no
```
Rembember to change out username and X with your own username and the server number. Type your password and press enter. Now you should be able to open your browser and paste in the second link from above, changing out the port number to the one you are using on your local machine.

```
http://localhost:1234/?token=a18268c5f164e0dbe2e52aec3b0429ee20264ee6828a2802
```
If you did everything correctly you now have access to all the notebooks in your browser and can start working!