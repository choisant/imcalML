## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0
## Name the analysis
#PBS -N XGBoost_hyperparams
## Choose queue
#PBS -q unlimited
. /home/agrefsru/.bashrc
cd /disk/atlas3/users/agrefsru/imcalML
conda activate xgboost

python ./src/XGBoost_hyperparams.py
  