## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0
## Name the analysis
#PBS -N p_value
## Choose queue
#PBS -q unlimited
. /home/agrefsru/.bashrc
cd /disk/atlas3/users/agrefsru/imcalML
conda activate imcal

python ./src/p_value_analysis.py
