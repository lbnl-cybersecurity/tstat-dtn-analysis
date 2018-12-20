Data analysis and prediction module on tstat flow data.
Module predicts the number of retransmitted packets on individual flows incorporating
the following factors per flow: tcp_rtt_avg', 'throughput_Mbps', 'duration'.
Module operates on timeseries smoothed (window =100) data.
Prediction is based on random forest regression.

The program requires a csv file with tstat records as input in the data analysis functions.
	Mandatory fields: 'percent_retrans', 'tcp_rtt_avg', 'throughput_Mbps', 'duration'

Available actions:
	Smooth -- perform smoothing (i.e. remove noise from to-be-analysed data)
	rf 	   -- perform random forest regrression over data
	across -- compare accuracy of prediction across different testing and training datasets

Functions:
	k_fold_random_forest_DD()
	Input:
		df : dataframe containing data
		n_fold: number of folds for cross validation
		var_2_pred: variable to predict
		predictor_var: table containing the variables that the prediction will be based on
	Returns:
		all_accuracy: registered accuracy of each fold

	across_datasets__random_forest_DD()
	Input:
		df_train: dataframe containing training data
		df_test: dataframe containing testing data
		var_2_pred: variable to predict
		predictor_var: table containing the variables that the prediction will be based on
	Returns:
		r2_score_kf: R2 score over the two datasets

	smooth_data_DD()
	Input:
		df : dataframe containing data
		features: selected features for smoothing
		window_size: data points that participate in each smooting operation
	Returns:
		df_new: dataframe containing smoothed data




