Client-server module for predicting percentage of retransmitted packets on individual network flows. 
The module incorporates random forest regression in order to predict the number of retransmitted packets on each flow.
The prediction algorithm operates on timeseries data and the input factors for each flow are:
["file_size_MB",'throughput_Mbps','duration', 'tcp_initial_cwin','src_ip','dst_ip', 'tcp_rtt_avg']

tstat data must be stored in a dataframe containing the following fields per flow:
['@timestamp','percent_retrans', "file_size_MB",'throughput_Mbps','duration','num_packets','packets_per_second','tcp_cwin_min','tcp_initial_cwin','tcp_max_seg_size','tcp_min_seg_size','tcp_mss','tcp_out_seq_pkts','tcp_pkts_dup','tcp_pkts_fc','tcp_pkts_fs','tcp_pkts_reor','tcp_pkts_rto','tcp_pkts_unfs','tcp_pkts_unk','tcp_pkts_unrto','tcp_rexmit_pkts','tcp_rtt_max','tcp_rtt_avg','tcp_rtt_std','tcp_win_max','tcp_win_min', 'tcp_cwin_max','src_ip','dst_ip','src_port', 'start']




convert_ips(file):
	file: file where the tstat data is stored 
	This function is responsible for converting IPs (minus the last octet) to numerical values so they can later 
	be processed by the regression (i.e. prediction function).
	If instead of an IP a hostname is provided then a DNS resolution is performed.

	returns: results are stored in a pkl file in a data frame manner

train(df, number_of_folds, prediction_var, prediction_fields):
	Generic regression prediction function based on Random Forest Regressor 

	df: dataframe where all tstat data are stored
	number_of_folds: number of validation times for the prediction algorithm
	prediction_var: variable to be predicted
	prediction_fields: variables participating in the prediction process

	Function separates dataset in testing and training subsets, builds the prediction model
	based on training data set and test the model using testing dataset

	Model is stored for later analysis 
	returns: prediction values

test(model, df, test_var, test_fields):
	testing function for an existing model 
	on new dataset




Running the prediction model:

This code assumes that the file with the tstat data is present.
In order to run:
1. Start server in pred_model.py : ./pred_model.py
2. Send request through client.py:
	- test for testing an existing model
	- train for training a new one