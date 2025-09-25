## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0
## Name the analysis
#PBS -N resnet_ensemble
## Choose queue
#PBS -q unlimited
. /home/agrefsru/.bashrc
cd /disk/atlas3/users/agrefsru/imcalML
conda activate imcal

python ./src/train_resnet_ensemble.py
