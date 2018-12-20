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



def main():
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_address = ('localhost', 10000)
	sock.connect(server_address)

	message = 'name_of_file ' + 'test'
	sock.sendall(message)
	while True:
		data = sock.recv(1)
		
		print(data)
		


if __name__ == "__main__":
    main()