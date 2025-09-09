import os
import h5py
import uproot as ur
import awkward as ak
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss as log_loss

from timeit import default_timer as timer
from datetime import timedelta

### PARAMETERS ###

HYPERPARAM_DICT = {
    "lr" : [0.01, 0.1],
    "max_depth" : [1, 2, 3, 4],
    "gamma" : [0, 0.1, 1],
    "lambda" : [0, 0.1, 1],
    "alpha" : [0, 0.1, 1],
    "subsample_f" : [0.25, 0.5, 0.75, 1],
    "colsample_f" : [0.25, 0.5, 0.75, 1]
}

DATA_TYPE = "high-level"
CUT = True
BH_LABELS = ["BH_n4_M8", "BH_n2_M10", "BH_n4_M10", "BH_n6_M10", "BH_n4_M12"]
LABELS = ["PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL"] + BH_LABELS
PLOT_LABELS = ["SPH_9", "BH_n4_M8", "BH_n2_M10", "BH_n4_M10", "BH_n6_M10", "BH_n4_M12"]

#Sets the algorithm type based on the number of classes
if (len(LABELS) > 2):
    ALGORITHM_TYPE = "multiclass-classification"
elif (len(LABELS) == 2):
    ALGORITHM_TYPE = "binary-classification"
else:
    print("Please input more than one label.")

if CUT:
    TRAIN_N_EVENTS = [60000, 60000, 24000, 25000, 26000, 20000]
    VAL_N_EVENTS = [20000, 15000, 6000, 15000, 7000, 15000]
    TEST_N_EVENTS = [85000, 100000, 29000, 32000, 36000, 25000]
    N_MIN = 5
    ST_MIN = 7 #TeV
    ETA_MAX = 2.4
    PT_MIN = 70 #GeV
    BH_TRAIN_CUT_DATAPATHS = [f"/disk/atlas3/data_MC/2dhistograms/BH/50/{label}_res50_STmin7_Nmin5_10000_events.h5" for label in BH_LABELS] 
    TRAIN_CUT_ID_DATAPATHS = ["/disk/atlas3/data_MC/2dhistograms/sph/50/PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL_res50_STmin7_Nmin5_10000_events.h5"] + BH_TRAIN_CUT_DATAPATHS
    
    BH_VAL_CUT_DATAPATHS = [f"/disk/atlas3/data_MC/2dhistograms/BH/50/{label}_test_res50_STmin7_Nmin5_3000_events.h5" for label in BH_LABELS]
    VAL_CUT_ID_DATAPATHS = ["/disk/atlas3/data_MC/2dhistograms/sph/50/PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL_test_res50_STmin7_Nmin5_3000_events.h5"] + BH_VAL_CUT_DATAPATHS
    
    BH_TEST_CUT_DATAPATHS = [f"/disk/atlas3/data_MC/2dhistograms/BH/50/{label}_test_res50_STmin7_Nmin5_15000_events.h5" for label in BH_LABELS]
    TEST_CUT_ID_DATAPATHS = ["/disk/atlas3/data_MC/2dhistograms/sph/50/PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL_test_res50_STmin7_Nmin5_15000_events.h5"] + BH_TEST_CUT_DATAPATHS

else:
    TRAIN_N_EVENTS = [10000]*len(LABELS)
    VAL_N_EVENTS = [3000]*len(LABELS)
    TEST_N_EVENTS = [3000]*len(LABELS)

DATA_PATH = "/disk/atlas3/data_MC/delphes/"

#Set data paths

TRAIN_FILENAMES = [f"{label}_{n}events.root" for label, n in zip(LABELS, TRAIN_N_EVENTS)]
VAL_FILENAMES = [f"{label}_{n}events.root" for label, n in zip(LABELS, VAL_N_EVENTS)]
TEST_FILENAMES = [f"{label}_{n}events.root" for label, n in zip(LABELS, TEST_N_EVENTS)]

TRAIN_DATAPATHS = [f"{DATA_PATH}/{TRAIN_FILENAME}" for TRAIN_FILENAME in TRAIN_FILENAMES]
VAL_DATAPATHS = [f"{DATA_PATH}/{VAL_FILENAME}" for VAL_FILENAME in VAL_FILENAMES]
TEST_DATAPATHS = [f"{DATA_PATH}/{TEST_FILENAME}" for TEST_FILENAME in TEST_FILENAMES]

# Save results to this path 
label_string = ""
for label in PLOT_LABELS:
    label_string += str(f"{label}_")
SAVE_PATH = f"./results/xgboost_hyperparams/{DATA_TYPE}_{label_string}CUT.csv"

### FUNCTIONS ###
def load_cut_event_numbers(hdf5file):
    with h5py.File(hdf5file, 'r') as f:
        keys = list(f.keys())
        data = [f[key]["event_id"] for key in keys]
        #create array
        ids = np.array(data).flatten()
        ids = ids.tolist()
        ids = [int(item[0]) for item in ids]
    return ids

def algorithm_definition(algorithm_type, labels):
    """ 
    Algorithm type defining function. Parameters: algorithm_type: ('binary-classification', 'multiclass-classification').
    """
    
    if algorithm_type == 'binary-classification':
        backgrounds = labels
        print("Binary-classification initialized.")

    elif algorithm_type == 'multiclass-classification':
        backgrounds = labels[1:]
        print("Multiclass-classification initialized.")
    else:
        raise Exception("Unsupported type of an algorithm. Only 'binary-classification' or 'multiclass-classification' allowed.")
    
    return algorithm_type, backgrounds, labels

def objective_metric_definition(algorithm_type):
    """ 
    Objective and metric definition function. Parameters: algorithm_type: ('regression', 'binary-classification', 'multiclass-classification').
    """

    if algorithm_type == "binary-classification":
        objective = "binary:logistic"                   # Possibilities: binary:logistic, binary:logitraw, binary:hinge
        metric = "logloss"                              # Possibilities: logloss, error, error@t, auc

    elif algorithm_type == "multiclass-classification":
        objective = "multi:softprob"                     # Possibilities: multi:softmax, multi:softprob
        metric = "mlogloss"                             # Possibilities: mlogloss, merror, auc
        
    else:
        raise Exception("Unsupported type of an algorithm. Only 'binary-classification' or 'multiclass-classification' allowed.")
    
    return objective, metric

def split_data_label(algorithm_type, df, split=False):
    
    if (algorithm_type == 'binary-classification') or (algorithm_type == 'multiclass-classification'):
        y = df['class'] # Class label
        X = df.drop(['class', "EventID"], axis = 1) # Drop class label and eventID
    
    if split==True:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size = 1/5, 
        )
        return X_train, X_test, y_train, y_test
    else:
        return X, y

def load_high_level(paths, n_events=[-1, -1], id_paths=None):
    """"
    Loads high level features from the root file.
    """
    keys = ["Event.Number", "Jet.Eta", "Jet.Phi", "Jet.PT", 
            "Muon.Charge", "Muon.Eta", "Muon.Phi", "Muon.PT", 
            "Electron.Charge", "Electron.Eta", "Electron.Phi", "Electron.PT", 
            "MissingET.MET", "MissingET.Eta", "MissingET.Phi"]
    ar = []
  # ---File Opening---
    for i, path in enumerate(paths):
        with ur.open(path) as temp_file:
                tree = temp_file['Delphes']
                ar_temp = tree.arrays(keys, library='ak')[0:n_events[i]]
        ar_temp['class']=i
        if CUT:
            event_ids = load_cut_event_numbers(id_paths[i])
            print(f"Events before cut: {len(ar_temp)}")
            array_event_ids = ak.to_numpy(ar_temp["Event.Number"]).flatten()
            matching_ids = np.argwhere(np.isin(array_event_ids, event_ids)).ravel()
            ar_temp = ar_temp[matching_ids]
            print(f"Events after cut: {len(ar_temp)}")
            ar = ak.concatenate([ar, ar_temp], axis=0)
        else:
            ar = ak.concatenate([ar, ar_temp], axis=0)
        print(f"Number of events loaded: {len(ar)}")

    
    # ---Custom Parameters---
    n_max_jets = 8                   # Number of most energetic particles to consider
    # ---Custom Parameters---
    jet_feats = ["Jet.Eta", "Jet.Phi", "Jet.PT"]
    for feat in jet_feats:
        ar[feat] = ak.pad_none(ar[feat], target = n_max_jets, clip = True)   # Padding
        for i in range(n_max_jets):
            ar[f"{feat}_{i}"] = ar[feat][:,i]                           # Adding a new feature on the i'th element of var-length features
        ar = ar[[x for x in ak.fields(ar) if x != feat]]              

    n_max_leptons = 2
    lepton_feats = ["Muon.Charge", "Muon.Eta", "Muon.Phi", "Muon.PT", "Electron.Charge", "Electron.Eta", "Electron.Phi", "Electron.PT"]
    
    for feat in lepton_feats:
        ar[feat] = ak.pad_none(ar[feat], target = n_max_leptons, clip = True)   # Padding
        for i in range(n_max_leptons):
            ar[f"{feat}_{i}"] = ar[feat][:,i]                           # Adding a new feature on the i'th element of var-length features
        ar = ar[[x for x in ak.fields(ar) if x != feat]]              
    
    met_feats = ["MissingET.MET", "MissingET.Eta", "MissingET.Phi"]
    n_max_met = 1
    # ---Flattening Execution---
    for feat in met_feats:
        ar[feat] = ak.pad_none(ar[feat], target = n_max_met, clip = True)   # Padding
        for i in range(n_max_met):
            ar[f"{feat}_{i}"] = ar[feat][:,i]                           # Adding a new feature on the i'th element of var-length features
        ar = ar[[x for x in ak.fields(ar) if x != feat]]                # Removing jagged arrays
    # ---Conversion to Pandas---
    #Rename
    ar["EventID"] = ar["Event.Number"]
    df = ak.to_dataframe(ar)
     # ---Data Cleaning Execution---
    discard = ["ST", "N", "Event.Number"] 
    for item in discard:
        df = df[df.columns.drop(list(df.filter(regex=item)))]
    return df

def load_low_level(paths, n_events=[-1, -1], id_paths=None, n_features:int=10):
    """
    Loads low-level data from calorimeters and the tracking system.
    """   

    keys = ["Event.Number", "Tower.Eta", "Tower.Phi", "Tower.Eem", "Tower.Ehad", "Track.PT", "Track.Eta", "Track.Phi"]
    ar = []

    # ---File Opening---
    for i, path in enumerate(paths):
        with ur.open(path) as temp_file:
                tree = temp_file['Delphes']
                ar_temp = tree.arrays(keys, library='ak')[0:n_events[i]]
        ar_temp['class']=i
        if CUT:
            event_ids = load_cut_event_numbers(id_paths[i])
            print(f"Events before cut: {len(ar_temp)}")
            array_event_ids = ak.to_numpy(ar_temp["Event.Number"]).flatten()
            matching_ids = np.argwhere(np.isin(array_event_ids, event_ids)).ravel()
            ar_temp = ar_temp[matching_ids]
            print(f"Events after cut: {len(ar_temp)}")
        ar = ak.concatenate([ar, ar_temp], axis=0)
    # ---Custom Parameters---
    n_max = n_features    # Number of most energetic particles to consider
    # Cuts
    
    # ---Rename---
    ar["Ehad"] = ar["Tower.Ehad"]
    ar["Ehad.Eta"] = ar["Tower.Eta"]
    ar["Ehad.Phi"] = ar["Tower.Phi"]
    ar["Eem"] = ar["Tower.Eem"]
    ar["Eem.Eta"] = ar["Tower.Eta"]
    ar["Eem.Phi"] = ar["Tower.Phi"]
    ar["EventID"] = ar["Event.Number"]

    # ---Sort---
    ehad_idx = ak.argsort(ar["Ehad"], ascending=False, axis=-1)
    ar["Ehad"] = ar["Ehad"][ehad_idx]
    ar["Ehad.Eta"] = ar["Ehad.Eta"][ehad_idx]
    ar["Ehad.Phi"] = ar["Ehad.Phi"][ehad_idx]
    eem_idx = ak.argsort(ar["Eem"], ascending=False, axis=-1)
    ar["Eem"] = ar["Eem"][eem_idx]
    ar["Eem.Eta"] = ar["Eem.Eta"][eem_idx]
    ar["Eem.Phi"] = ar["Eem.Phi"][eem_idx]
    track_idx = ak.argsort(ar["Track.PT"], ascending=False, axis=-1)
    ar["Track.PT"] = ar["Track.PT"][track_idx]
    ar["Track.Eta"] = ar["Track.Eta"][track_idx]
    ar["Track.Phi"] = ar["Track.Phi"][track_idx]

    # ---Flattening---
    var_list = []
    for feat in ak.fields(ar):
        if (feat != "EventID"):
            try:                                        # Attempting to flatten features - catches all variable length features
                ak.flatten(ar[str(feat)],axis = 1)
                var_list.append(str(feat))              # Fields which need to be flattened and padded
            except Exception:
                ar = ar

    for feat in var_list:
        ar[feat] = ak.pad_none(ar[feat], target = n_max, clip = True)   # Padding
        for i in range(n_max):
            ar[f"{feat}_{i}"] = ar[feat][:,i]                           # Adding a new feature on the i'th element of var-length features
        ar = ar[[x for x in ak.fields(ar) if x != feat]]                # Removing jagged arrays

    # ---Conversion to Pandas---
    df = ak.to_dataframe(ar)

    # ---Data Cleaning Execution---
    discard = ["Tower.Eta", "Tower.Phi", "Tower.Eem", "Tower.Ehad", "Event.Number", "Jet.Eta", "Jet.PT", "Muon.PT", "Muon.Eta",
            "Electron.PT", "Electron.Eta", "MissingET.MET", "ST", "N"] 
    for item in discard:
        df = df[df.columns.drop(list(df.filter(regex=item)))]
    return df

def create_hyperparam_df(hyperparam_dict):
    # Total number of combinations of hyperparams
    n_var_hyperparams = 1
    for key in hyperparam_dict.keys():
        n_var_hyperparams = n_var_hyperparams*len(hyperparam_dict[key])
    # Create dataframe
    df = pd.DataFrame(columns=hyperparam_dict.keys())
    # With all permutations
    iter_perms_down = n_var_hyperparams
    iter_perms_up = 1
    for key in hyperparam_dict.keys():
        values = hyperparam_dict[key]
        iter_perms_down = int(iter_perms_down/len(values))
        df[key] = np.array([[x]*iter_perms_down for x in values]*iter_perms_up).flatten()
        iter_perms_up = int(iter_perms_up*len(values))

    return df

### LOAD DATA ###
if CUT:
    train_df = load_high_level(TRAIN_DATAPATHS, TRAIN_N_EVENTS, TRAIN_CUT_ID_DATAPATHS).droplevel('subentry')
    val_df = load_high_level(VAL_DATAPATHS, VAL_N_EVENTS, VAL_CUT_ID_DATAPATHS).droplevel('subentry')
    test_df = load_high_level(TEST_DATAPATHS, TEST_N_EVENTS, TEST_CUT_ID_DATAPATHS).droplevel('subentry')
else:
    train_df = load_high_level(TRAIN_DATAPATHS, TRAIN_N_EVENTS)
    val_df = load_high_level(VAL_DATAPATHS, VAL_N_EVENTS)

# Shuffle train data
train_df.sample(frac=1)

### PREPARE XGBOOST ###

algorithm_type, backgrounds, labels = algorithm_definition(ALGORITHM_TYPE, PLOT_LABELS)
objective, metric = objective_metric_definition(algorithm_type)

# Split input and label
X_train, y_train = split_data_label(algorithm_type, train_df, split=False)
X_val, y_val = split_data_label(algorithm_type, val_df, split=False)

### CREATE HYPERPARAM DATAFRAME ###

gridsearch_df = create_hyperparam_df(HYPERPARAM_DICT)
gridsearch_df["ACC"] = np.zeros(len(gridsearch_df))
gridsearch_df["LogLoss"] = np.zeros(len(gridsearch_df))
for label in PLOT_LABELS:
    gridsearch_df[f"ACC_{label}"] = np.zeros(len(gridsearch_df))
gridsearch_df["n_trees"] = np.zeros(len(gridsearch_df))
gridsearch_df["Training time"] = ["time"]*len(gridsearch_df)
gridsearch_df["Time"] = np.zeros(len(gridsearch_df))
gridsearch_df["ACC_train"] = np.zeros(len(gridsearch_df))
gridsearch_df["ACC_val"] = np.zeros(len(gridsearch_df))
gridsearch_df["Loss_train"] = np.zeros(len(gridsearch_df))
gridsearch_df["Loss_val"] = np.zeros(len(gridsearch_df))
gridsearch_df.to_csv(f"{SAVE_PATH}")

### START GRID SEARCH ###
for i in range(len(gridsearch_df)):
    start_train = timer()
    print(f"Gridpoint {i} out of {len(gridsearch_df)}")
    # Hyperparameters
    hyperparameters = {
        # ---General Parameters---
        'booster':'gbtree',                     # type of model to run at each iteration

        # ---Booster Parameters---
        'n_estimators':10000,                    # number of classifiers
        'learning_rate':gridsearch_df["lr"][i],                   # learning rate
        'max_depth':int(gridsearch_df["max_depth"][i]),                          # the maximum depth of a tree
        'min_child_weight':1,                   # defines the minimum sum of weights in a child
        'gamma':gridsearch_df["gamma"][i],                              # specifies minimum loss reduction to make a split
        'subsample':gridsearch_df["subsample_f"][i],                        # defines random fraction of observations for each tree
        'colsample_bytree':gridsearch_df["colsample_f"][i],                   # defines random fraction of columns for each tree
        'reg_alpha':gridsearch_df["alpha"][i],                          # L1 regularization term on weights
        'reg_lambda':gridsearch_df["lambda"][i],                         # L2 regularization term on weights
        'max_delta_step':0,                     # tree’s weight estimation
        'early_stopping_rounds':10,

        # ---Learning Task Parameters---
        'objective':objective,                  # defines loss function to be minimized
        'eval_metric':metric,
        'tree_method':'gpu_hist',               # tree constructing method. Possibilities: hist, gpu_hist, approx, exact
        'gpu_id':1,                             # selects which GPU card to use (uncheck in case of not using gpu)
        #'seed':1,                               # seed statistic number

        # ---Multiclass Parameters---
        'num_class':len(np.unique(y_train))         # number of classes (for multiclass-classification only)
    }

    if algorithm_type != 'multiclass-classification':
        del hyperparameters['num_class']

    fit_parameters= {
        'X':X_train,
        'y':y_train,
        "eval_set" : [(X_train, y_train), (X_val, y_val)],
        'verbose':0
    }

    # Create classifier object
    if (algorithm_type == 'binary-classification') or (algorithm_type == 'multiclass-classification'):
        xgb_model = xgb.XGBClassifier(**hyperparameters)
    else:
        raise Exception("Unsupported type of an algorithm. Only 'regression', 'binary-classification' or 'multiclass-classification' allowed.")

    # Fit model
    xgb_model.fit(**fit_parameters)

    # Test model
    train_predictions = xgb_model.predict(
        X_train, 
        iteration_range=[0, xgb_model.best_ntree_limit]
    )
    val_predictions = xgb_model.predict(
        X_val, 
        iteration_range=[0, xgb_model.best_ntree_limit]
    )
    #Get a new subset each time
    X_test, X_safe, y_test, y_safe = split_data_label(algorithm_type, test_df, split=True)
    predictions = xgb_model.predict(
        X_test, 
        iteration_range=[0, xgb_model.best_ntree_limit]
    )

    predictions_proba = xgb_model.predict_proba(
        X_test,
        iteration_range=[0, xgb_model.best_ntree_limit]
    )

    df_results = pd.DataFrame(columns=["Truth", "Prediction", "Confidence", "Correct"])
    df_results["Truth"] = y_test
    df_results["Prediction"] = predictions
    df_results["Confidence"] = np.max(predictions_proba, axis=-1)
    for j in range(len(PLOT_LABELS)):
        df_results[f"{j}"] = predictions_proba[:,j]
    correct_list = np.array([True]*len(df_results))
    correct_list[df_results["Prediction"] != df_results["Truth"]] = False
    df_results["Correct"] = correct_list

    gridsearch_df.loc[i, "ACC"] = accuracy_score(df_results["Truth"], df_results["Prediction"], normalize=True)
    gridsearch_df.loc[i, "LogLoss"] = log_loss(df_results["Truth"], df_results[[f"{j}" for j in range(len(LABELS))]], normalize=True)
    cf_matrix = confusion_matrix(df_results["Truth"], df_results["Prediction"], normalize="true")
    for j in range(len(PLOT_LABELS)):
        gridsearch_df.loc[i, f"ACC_{PLOT_LABELS[j]}"] = cf_matrix[j,j]
    gridsearch_df.loc[i, "n_trees"] = xgb_model.best_ntree_limit
    end_train = timer()
    total_time = timedelta(seconds=end_train-start_train)
    gridsearch_df.loc[i, "n_trees"] = xgb_model.best_ntree_limit
    gridsearch_df.loc[i, "Training time"] = str(total_time)
    gridsearch_df.loc[i, "Time"] = end_train

    #Check for overtraining
    gridsearch_df.loc[i, "ACC_train"] = accuracy_score(y_train, train_predictions)
    gridsearch_df.loc[i, "ACC_val"] = accuracy_score(y_val, train_predictions)
    gridsearch_df.loc[i, "Loss_train"] = xgb_model.evals_result()["validation_0"]['mlogloss'][-1]
    gridsearch_df.loc[i, "Loss_val"] = xgb_model.evals_result()["validation_1"]['mlogloss'][-1]

    ### SAVE RESULTS EVERY ITERATION ###
    gridsearch_df.to_csv(f"{SAVE_PATH}", mode='w')