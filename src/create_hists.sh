## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0
## Name the analysis
#PBS -N create_hists
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe
## Array of jobs
#PBS -t 1-6

. /home/agrefsru/.bashrc
cd /disk/atlas3/users/agrefsru/imcalML
conda activate imcal

RES=200
ST_min=7
Nmin=5
MODE="validation"

SCRIPT="src/root_to_2dhists_cuts.py"
DATA_PATH="/disk/atlas3/data_MC/delphes/"
LABELS=("PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL" "BH_n4_M8" "BH_n2_M10" "BH_n4_M10" "BH_n6_M10" "BH_n4_M12")
FOLDERS=("sph" "BH" "BH" "BH" "BH" "BH")



if [ "${MODE}" == "training" ]; then
    N_EVENTS=10000
    DELPHES_EVENTS=(60000 60000 24000 25000 26000 20000)
elif [ "${MODE}" == "validation" ]; then
    N_EVENTS=3000
    DELPHES_EVENTS=(20000 15000 6000 15000 7000 15000)
elif [ "${MODE}" == "testing" ]; then
    N_EVENTS=15000
    DELPHES_EVENTS=(85000 100000 29000 32000 36000 25000)
else
    echo "Choose a valid MODE!"
    exit 1
fi


### for each job ###
LABEL=${LABELS[${PBS_ARRAYID}-1]}
EVENTS=${DELPHES_EVENTS[${PBS_ARRAYID}-1]}
LOAD_PATH="${DATA_PATH}${LABEL}_${EVENTS}events.root"
SAVE_PATH="/disk/atlas3/data_MC/2dhistograms/${FOLDERS[${PBS_ARRAYID}-1]}/${RES}"

if [ "${MODE}" == "training" ]; then
    FILENAME="${LABEL}"
elif [ "${MODE}" == "validation" ]; then
    FILENAME="${LABEL}_test"
elif [ "${MODE}" == "testing" ]; then
    FILENAME="${LABEL}_test"
else
    echo "Choose a valid MODE!"
    exit 1
fi
echo $EVENTS
echo $LOAD_PATH
echo $FILENAME

python $SCRIPT -f $LOAD_PATH -s $SAVE_PATH --ST_min $ST_min --N_min $Nmin -n $FILENAME -r $RES -N $N_EVENTS
