#!/usr/bin/env python

import requests
import socket
from elasticsearch import Elasticsearch, helpers
import json
import csv
import sys
import os
import re
import ipaddress
from itertools import izip, count
import datetime
import dateutil.relativedelta
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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
from hurry.filesize import size
import pandas as pd
import pickle



def convert_ips(file):
	df = pd.read_pickle(file)
	for (i,j) in izip(df.src_ip,df.dst_ip):
		print(i,j)
		try:
			address = int(ipaddress.IPv4Address(unicode(socket.gethostbyname(i),'utf-8')))
			address2 = int(ipaddress.IPv4Address(unicode(socket.gethostbyname(j),'utf-8')))

		except socket.gaierror:
			print('oops\n')
			print(i)
			print(j)
			print('I will try and see if there is an IP included in the hostname string\n')
			ip = re.compile('(([2][5][0-5]\.)|([2][0-4][0-9]\.)|([0-1]?[0-9]?[0-9]\.)){3}'+'(([2][5][0-5])|([2][0-4][0-9])|([0-1]?[0-9]?[0-9]))')
			match = ip.search(i)
			match2 = ip.search(j)
			if match:
				print(match.group())
				i = str(match.group())
				
			elif match2:
				print(match2.group())
				j = str(match2.group())
				
			else:
				print('Cannot resolve or locate an IP substring\n')
				with open('problems.txt', 'a') as f:
					f.write(str(i)+ ' '+str(j)+'\n')

					
			df = df[df.src_ip != i]
			df = df[df.dst_ip != j]
	
	df['converted_sips'] = pd.Series(df['src_ip'].apply(lambda x: int(ipaddress.IPv4Address(unicode(socket.gethostbyname(x),'utf-8')))), index = df.index)
	df['converted_dips'] = pd.Series(df['dst_ip'].apply(lambda x: int(ipaddress.IPv4Address(unicode(socket.gethostbyname(x),'utf-8')))), index = df.index)
	df.to_pickle('ips.p')

def train(df, number_of_folds, prediction_var, prediction_fields):
	folds = number_of_folds
	X_index = np.random.randint(df.shape[0], size = (df.shape[0],1))
	print("The index is \n")
	print(X_index) 


	kf = KFold(n_splits = folds)
	print(kf.get_n_splits(X_index))
	KFold(n_splits = folds, random_state=None, shuffle=False)
	all_accuracy = []
	for train_index, test_index in kf.split(X_index):
	    X_train, X_test = X_index[train_index], X_index[test_index]
	    X_train = np.reshape(X_train, np.size(X_train))
	    X_test = np.reshape(X_test, np.size(X_test))
	    train = df.iloc[X_train[:],:]
	    print(X_test)
	    print(df.shape)
	    test = df.iloc[X_test[:],:]
	    ytrain = train[prediction_var]
	
	    ytest = test[prediction_var]

	    

	    Xtrain = train[prediction_fields]
	    Xtest = test[prediction_fields]
	    print(Xtrain)
	    
	    rfr = RandomForestRegressor(n_estimators = 100, verbose = True)
	    regr = rfr.fit(Xtrain,ytrain)

	    r2_score_kf = r2_score(ytest, regr.predict(Xtest))
	    all_accuracy.append(r2_score_kf)
	    print(r2_score_kf)
	print(all_accuracy)
	model_file = 'trained_model.pkl'
	pickle.dump(regr, open(model_file, 'wb'))
	return regr.predict(Xtest)

def test(model, df, test_var, test_fields):
	ytest = df[test_var]
	Xtest = df[test_fields]

	r2_score_kf = r2_score(ytest, model.predict(Xtest))
	print(r2_score_kf)
	mae = sklearn.metrics.mean_absolute_error(ytest, model.predict(Xtest))
	res = model.predict(Xtest)
	print(res)

	return model.predict(Xtest)

def main():

	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_address = ('localhost', 10000)
	sock.bind(server_address)
	sock.listen(1)

	while True:
		connection, client_address = sock.accept()
		data = connection.recv(1024)
		if len(data) is not None:
			print('This is the file name with the data \n')
			print(data.split(' ')[0])
			print('I will now check if the file exists\n')
			#try:
			os.path.isfile(data.split()[0])
			print('The file exists\n')

			if data.split()[1] == 'retrain':
				print('You have chosen to retrain the packet loss prediction module\n')
	
				if(os.path.isfile('ips.p')):
					print('file is here\n')
					print(data.split()[0]+'_ips.pkl')
				else:
					print('The ips are not converted to numerical values I will have to convert them\n')
					convert_ips(data.split()[0])

				df = pd.read_pickle('ips.p')	
				df = df.rolling(100, win_type ='triang').mean()
				df = df.dropna(how='any')	

				retrans = train(df, 10, ['percent_retrans'], ['tcp_rtt_avg', 'file_size_MB', 'throughput_Mbps', 'duration', 'converted_sips','converted_dips','tcp_initial_cwin'])
				print(retrans)
				


			elif(data.split()[1] == 'test'):
				print('You have selected testing on new data with an existing model\n')
				#try:
				os.path.isfile('trained_model.pkl')
				model = pickle.load(open('trained_model.pkl', 'rb'))
				if (os.path.isfile('ips_test.p')):
					print('The testing file with the converted IPs is here\n')
				else:
					convert_ips(data.split()[0])	
				df_test = pd.read_pickle('ips_test.p')	
				df_test = df_test.rolling(100, win_type ='triang').mean()
				df_test = df_test.dropna(how='any')	
				df_dtn01 = df_test[df_test.dst_ip=='XXX.nersc.gov']
				str_dtn = ['dtn','nersc.gov']

				for i in df_test.dst_ip.unique():
					if all(x in i for x in str_dtn):
						print(i)
						df_dtn01 = df_test[df_test.dst_ip==i]

						df_dtn01.reset_index(drop=True)
						retrans = test(model, df_dtn01,['percent_retrans'], ['tcp_rtt_avg', 'file_size_MB', 'throughput_Mbps', 'duration', 'converted_sips','converted_dips','tcp_initial_cwin'] )	

						print(np.array(retrans).size)

				
						new_df = pd.concat([pd.to_datetime(df_dtn01['@timestamp'],infer_datetime_format=True),df_dtn01['percent_retrans'], df_dtn01['src_ip']],axis=1, keys = ['timestamp', 'original_retrans','src_ip'] )
						new_df['predicted_retrans'] = retrans
						print(new_df)
						new_df.to_csv('res_'+str(i)+'.csv',index=None, sep=' ', mode='a')
				#np.savetxt('res.out',np.array(retrans), delimiter = ',')

				#except:
				#	print('The model is not here, do you want to retrain?\n')
			connection.sendall('ok')



			# except:
			# 	print('The file is not here you need to call elastic\n')









if __name__ == "__main__":
    main()



