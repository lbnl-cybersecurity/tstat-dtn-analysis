import os
import sys
import json
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from  ggplot import *
import matplotlib.pyplot as plt
import random
import seaborn as sns
import scipy
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold # import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor


import requests
import socket
from elasticsearch import Elasticsearch, helpers
import json
import csv
import sys
import ipaddress
import itertools
import datetime
import dateutil.relativedelta
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from operator import itemgetter
from matplotlib.backends.backend_pdf import PdfPages



MAX_WINDOW = 10

def moving_average(data, _range):
    ret = np.cumsum(data, dtype=float)
    ret[_range:] = ret[_range:] - ret[:-_range]
    return ret[_range - 1:] / _range

def SMA(data, _range, slide):
    ret = moving_average(data, _range)[::slide]
    return list(ret)

def kurtosis(values):
    return scipy.stats.kurtosis(values)

def sucessive_slopes(vals):
    n = len(vals)
    diffs = []
    for i in range(n - 1):
        diffs.append((vals[i + 1] - vals[i]))
    return diffs

def roughness(vals):
    diff = sucessive_slopes(vals)
    return np.std(diff)


def smooth_simple(data, resolution=None):
    if resolution:
        paa_len = int(len(data) / resolution)
        if paa_len > 1:
            data = SMA(data, paa_len, paa_len)
    orig_kurt   = kurtosis(data)
    min_obj     = roughness(data)
    window_size = 1
    for w in range(2, int(len(data) / 10)):
        smoothed = SMA(data, w, 1)
        if kurtosis(smoothed) >= orig_kurt:
            r = roughness(smoothed)
            if r < min_obj:
                min_obj = r
                window_size = w
    return data, window_size

#### ---------------------------------------------------------------
    

def smooth_data_DD(df, features, window_size, plt_flag):
#     df = df.head(1000)
    df_new =  pd.DataFrame()
    df["idx"] = range(0, df.shape[0])
    
    for f in features:
        header = ["idx", f]
        new_df = df[["idx", f]]
        new_df.to_csv( f+".csv", columns = header,  index=False)
        raw_data = load_csv(f+".csv")
#         data, window_size = smooth_simple(raw_data,  resolution=100)
        smoothed_data = plot(raw_data, window_size, plt_flag)
        df_new[f] = smoothed_data
    return df_new
        

    
#### ---------------------------------------------------------------
    
def k_fold_random_forest_DD(df, n_fold, var_2_pred, predictor_var):

    all_accuracy = []

    X = np.random.randint(df.shape[0], size=(df.shape[0],1))
    kf = KFold(n_splits=n_fold) # Define the split - into 2 folds
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    KFold(n_splits=n_fold, random_state=None, shuffle=False)
    
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        X_train = np.reshape(X_train, np.size(X_train))
        X_test = np.reshape(X_test, np.size(X_test))
        train = df.iloc[X_train[:],:]
        test = df.iloc[X_test[:],:]
    
        ytrain = train[var_2_pred]
        ytest  = test[var_2_pred]
    
        Xtrain = train[predictor_var]
        Xtest  = test[predictor_var]

        rfr = RandomForestRegressor(n_estimators = 100, verbose = True)
        regr = rfr.fit(Xtrain,ytrain)
        r2_score_kf = r2_score(ytest, regr.predict(Xtest))
        all_accuracy.append(r2_score_kf)
    
    return all_accuracy   


#### ---------------------------------------------------------------


def across_datasets__random_forest_DD(df_train, df_test, var_2_pred, predictor_var):
    ytrain = df_train[var_2_pred]
    ytest  = df_test[var_2_pred]
    Xtrain = df_train[predictor_var]
    Xtest  = df_test[predictor_var]

    rfr = ExtraTreesRegressor(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    regr = rfr.fit(Xtrain,ytrain)
    r2_score_kf = r2_score(ytest, regr.predict(Xtest))
    return r2_score_kf


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--smooth", help="invoke this option if you like to apply smoothing over the dataset")
	parser.add_argument('--rf', help = 'this option is for random forest prediction')
	parser.add_argument("--across", help = 'invoke this option if you want to do cross dataset comparisson')
	parser.add_argument('--dataset', help = 'provide the path for the the data file')
	args = parser.parse_args()



	var_2_pred = ['percent_retrans']
	predictor_var = ['tcp_rtt_avg', 'throughput_Mbps', 'duration' ]
	n_fold = 10

	if args.dataset:
		print('Path of the data file is: %s'% args.dataset)
		df = pd.read_csv(args.dataset, sep='\t' )
		if args.rf:
			print('I will apply random forest prediction over the data\n')
			all_accuracy1 = k_fold_random_forest_DD(df, n_fold, var_2_pred, predictor_var)
			print('The accuracy over dataset %s is %s \n' % (args.dataset, all_accuracy1))
		elif args.across:
			print('I will apply random forest across datasets \n')
			all_accuracy2 = across_datasets__random_forest_DD(df, df,var_2_pred, predictor_var)
			print('Registered accuracies across datasets are: %s' % all_accuracy2)
		elif args.smooth:
			print('I will now proceed with the smoothing \n')
			features = ['percent_retrans', 'tcp_mss', 'throughput_Mbps', 'tcp_rtt_avg','duration', 'tcp_initial_cwin', 'tcp_win_max' ]
			window_size = 12
			df_new = smooth_data_DD(df, features, window_size, 1)


	else:
		print('You did not provide any dataset, I will exit now \n')
		exit(0)

if __name__== "__main__":
	main()



