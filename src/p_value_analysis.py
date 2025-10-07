#scientific libraries and plotting
import numpy as np
import scipy as scipy
from scipy import stats
import pandas as pd
import math
import os
import random
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


#variables controlled by the user. Change these to fit your specific needs.
RES = 50 #resolution
LABELS = ["PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL", "BH_n4_M8", "BH_n2_M10", "BH_n4_M10", "BH_n6_M10", "BH_n4_M12"]
PLOT_LABELS = ["SPH_9", "BH_n4_M8", "BH_n2_M10", "BH_n4_M10", "BH_n6_M10", "BH_n4_M12"]

CLASSES = len(LABELS) #The number of output nodes in the net, equal to the number of classes
FOLDERS = ["sph", "BH", "BH", "BH", "BH", "BH"]
TEST_N_EVENTS = 15000
CUT = True
N_max = 45 #Maximum number of events measured
M = 3000 #Number of experiments to average
STATISTIC = "binomial" # available statistics: ["poisson", "chi2"]

LOAD_FOLDER = "./results/predictions/resnet"
MODEL_NAME = "thesis_ensemble"
LOAD_PATH = f"{LOAD_FOLDER}/{MODEL_NAME}.csv"

####### STATISTICAL TESTS #########

#Chi2 test
def chi2(n_expected, n_data):
    return(np.sum((n_data-n_expected)**2/(n_expected)))

def chi2_method_pearson(prob_dist, data, classes, verbose=False):

    #Number of classified particles in each class
    N_samp = len(data)
    #Number of predicted events for theory
    N_expected = prob_dist*N_samp
    N_data = np.array([len(data[data["Prediction"]==i]) for i in range(classes)])
    #Calculate chi2_value
    chi2_value=chi2(N_expected, N_data)
    #Number of degrees of freedom
    ndof = classes-1
    #Get the p-value from the chi2
    p_value = np.float64((np.float64(1) - stats.chi2.cdf(chi2_value, ndof)))

    if verbose:
        #if p_value < 10**(-10):
        #print("p value: ", p_value)
        print(f"Histogram for data: {N_data}")
        print(f"Histogram theory: {N_expected}")
        print(f"The chi2 value for theory is {chi2_value}. The p-value is {p_value}")
    
    return p_value

#poisson test
def poisson(n, l):
    return((l**n)*np.exp(-l)/math.factorial(n))

def poisson_method(prob_dist, data, classes, verbose=False):
    #Number of classified particles in each class
    N_samp = len(data)
    #Number of predicted events for theory
    N_expected = prob_dist*N_samp
    N_data = np.array([len(data[data["Prediction"]==i]) for i in range(classes)])
    
    j_max = np.argmax(N_expected)
    lambda_j = N_expected[j_max]
    n_obs = N_data[j_max]
    p_value = np.sum([poisson(n, lambda_j) for n in range(0, n_obs+1)])

    if verbose:
        print(f"Histogram for data: {N_data}")
        print(f"Histogram theory: {N_expected}")
        print(f"The p value for theory is {p_value}")
        print(f"{lambda_j}")

    return p_value

# binomial test
def binomial(c, n, p):
    return((math.factorial(n)/(math.factorial(c)*math.factorial(n-c)))*(p**c)*((1-p)**(n-c)))

def binomial_method(hypothesis_index, prob_dist, data, classes, verbose=False):
    #Number of classified particles in each class
    N_samp = len(data)
    #Number of expected events
    N_exp = prob_dist*N_samp
    N_data = np.array([len(data[data["Prediction"]==i]) for i in range(classes)])
    n_obs = N_data[hypothesis_index]
    
    p_value = np.sum([binomial(n, N_samp, prob_dist[hypothesis_index]) for n in range(0, n_obs+1)])

    if verbose:
        print(f"Empirical frequencies: {N_data/N_samp}")
        print(f"Expected frequencies: {prob_dist}")
        print(f"The p value for theory is {p_value}")

    return p_value

####### Read datafile with prediction ########
df = pd.read_csv(LOAD_PATH)
datasets = [0]*CLASSES
for i in range(CLASSES):
    df_temp = df[df["Truth"] == i ]
    datasets[i] = df_temp

#Do this calculation just once
#Normalized prediction arrays for all datasets [hist1, hist2, ...] where hist1 = [a, b, c... d] with sum(hist1)=1
cf_matrix = confusion_matrix(df["Truth"].astype(int), df["Prediction"].astype(int), normalize="true")
p_estimated = np.array([row[i] for i, row in enumerate(cf_matrix)])


#get p-value as a function of number of events
if N_max == 1000:
    N_list = [10, 50, 100, 500, 1000]
else:
    N_list = np.arange(2, N_max, 1) #A list of integers from 2 to N_max

#RUN EXPERIMENT
df_results = [[0]*CLASSES for i in range(CLASSES)]
for i in tqdm(range(len(datasets))): #Iterate over all types of pseudodata
    for s, hypothesis in enumerate(cf_matrix): #Iterate over all hypothesis
        df_temp = pd.DataFrame(columns=["N", "average", "min", "max", "std", "5_percentile", "95_percentile"])
        df_temp["N"] = N_list
        avg_temp = np.zeros(len(N_list))
        min_temp = np.zeros(len(N_list))
        max_temp = np.zeros(len(N_list))
        std_temp = np.zeros(len(N_list))
        avg_temp = np.zeros(len(N_list))
        lower_percentile_temp = np.zeros(len(N_list))
        upper_percentile_temp = np.zeros(len(N_list))

        for j, n in enumerate(N_list): # Iterate over number of events detected
            rand_idx = [random.sample(range(len(datasets[0])-N_max), n) for k in range(0, M)]
            experiments = [datasets[i].iloc[rand_idx[k]] for k in range(0, M)] #Generate a list of M number of experiments with n events
            if STATISTIC=="poisson":
                results = np.array([poisson_method(hypothesis, experiment, CLASSES) for experiment in experiments])
            elif STATISTIC=="chi2":
                results = np.array([chi2_method_pearson(hypothesis, experiment, CLASSES) for experiment in experiments])
            elif STATISTIC=="binomial":
                results = np.array([binomial_method(s, hypothesis, experiment, CLASSES) for experiment in experiments])
            else:
                print("Choose an available statistic.")
                exit()
            avg_temp[j] = np.average(results, axis=0)
            min_temp[j] = results.min()
            max_temp[j] = results.max()
            std_temp[j] = np.std(results, axis=0)
            lower_percentile_temp[j] = np.percentile(results, 25)
            upper_percentile_temp[j] = np.percentile(results, 75)
        
        df_temp["average"] = avg_temp
        df_temp["min"] = min_temp
        df_temp["max"] = max_temp
        df_temp["std"] = std_temp
        df_temp["25_percentile"] = lower_percentile_temp
        df_temp["75_percentile"] = upper_percentile_temp

        df_results[i][s] = df_temp
# Save results
if (not os.path.isdir(f"./results/{STATISTIC}/{MODEL_NAME}") ):
        os.mkdir(f"./results/{STATISTIC}/{MODEL_NAME}")
for i in range(CLASSES):
    for j in range(CLASSES):
        df_results[i][j].to_csv(f"./results/{STATISTIC}/{MODEL_NAME}/m_{M}_truth_{LABELS[i]}_hypothesis_{LABELS[j]}.csv")