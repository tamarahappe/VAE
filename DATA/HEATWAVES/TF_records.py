import numpy as np 
from netCDF4 import Dataset
import pandas as pd
import os
import tensorflow as tf
import argparse

# supporting functions
def loading_heatwave_data(ensemble_number:int, path: str):
	'''To load in the heatwaves of respective ensemble member
	INPUTS:
	ensemble_number:int = number of ensemble member
	path:str = path where data is stored
	
	Takes in ensemble number and path where data is stored, returns lists with
	ensemble_i: list with ensemble member
	years_i: list with respective year
	day0_i: list with start of day of heatwave
	years_ind: index based on year and ensemble
	'''
	
	curdir = os.getcwd()
	os.chdir(f'{path}')
	file = f"westEU_JulyAugust_tas_h_day_NH_ensemble{ensemble_number}.csv"
	heatwaves_df = pd.read_csv(fr'{file}')

	ensemble = heatwaves_df["Ensemble member"]
	years = heatwaves_df["year"]
	day0 = heatwaves_df["day 0 (index)"]
	dates = heatwaves_df["start date"]
	years_n=100

	#retrieve heaywave event information
	ensemble_i, years_i, day0_i, years_ind, dates_i = [], [], [], [], []
	for i in range(len(years)):
		if day0[i] == " ": #discard noise
			continue
		years_i.append(float(years[i]))
		years_ind.append(float(str(years[i])[-1]) + ensemble[i]*10)
		ensemble_i.append(float(ensemble[i]))
		day0_i.append(float(day0[i]))
		dates_i.append(dates[i])
	os.chdir(curdir)
	return ensemble_i, years_i, day0_i, years_ind, dates_i


def preprocessing(variables, ensemble_member, path, years_n, extent_name, method):
	"""
	Purpose is to load data of each variable, to cut it to the desired extent,
	and to return the normalized/standardized values
	
	INPUTS
	variables: list of varariable names
	ensemble member: integer of ensemble member number 
	path: string of the path name where the data is stored
	years_n: integer of the number of years in the ensemble
	extent: list of 4 integers with [lon_min,lon_max,lat_min,lat_max]
	
	OUTPUTS
	outs: list of arrays, with each array containing one variable 
	
	"""
	outs = []
	if type(variables) != list:
		raise ValueError("variables is not in list format")

	curdir = os.getcwd()
	os.chdir(path)
	print(curdir, os.getcwd())
	
	######
	for var in variables:
		print(f"variable is {var}, ensemble is {ensemble_member}")
		if var == 'stream250':
			var_array = Dataset(fr"{var}/{extent_name}_june2sept_{var}_h_day_NH_ensemble{ensemble_member}.nc")["stream"][:] #load file
			var_lons = Dataset(fr"{var}/{extent_name}_june2sept_{var}_h_day_NH_ensemble{ensemble_member}.nc")["lon"][:]
			var_lats = Dataset(fr"{var}/{extent_name}_june2sept_{var}_h_day_NH_ensemble{ensemble_member}.nc")["lat"][:]
		elif var == 'rsds' or var == 'psl':
			var_array = Dataset(fr"{var}/{extent_name}_june2sept_{var}_h_day_NH_ensemble{ensemble_member}.nc")[var][:] #load file
			var_lons = Dataset(fr"{var}/{extent_name}_june2sept_{var}_h_day_NH_ensemble{ensemble_member}.nc")["lon"][:]
			var_lats = Dataset(fr"{var}/{extent_name}_june2sept_{var}_h_day_NH_ensemble{ensemble_member}.nc")["lat"][:]
			
		#standardization toepassen
		#standardization toepassen
		if method == 'standardization':
			print("standardization method: out=(X-Xmean)/(Xstd)")
			if var == 'stream250':
				var_mean = Dataset(fr"{var}/{extent_name}_june2sept_7dayrunningmean_{var}_train.nc")['stream'][:]
				var_std = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningstd_{var}_train.nc')['stream'][:]
			else:
				var_mean = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningmean_{var}_train.nc')[var][:]
				var_std = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningstd_{var}_train.nc')[var][:]
			#breakpoint()
			var_out= np.empty_like(var_array)
			for year in range(years_n): 
				var_year = (var_array[year*122:(year+1)*122,:,:] - var_mean) / var_std
				var_out[year*122:(year+1)*122,:,:]=var_year
			outs.append(var_out)

		elif method == 'RAW':
			print("NO STANDARDIZATION METHOD")
			outs.append(var_array)

		elif method == 'standardization_cut':
			print("standardization method: out=(X-Xmean)/(3 * Xstd)")
			if var == 'stream250':
				var_mean = Dataset(fr"{var}/{extent_name}_june2sept_7dayrunningmean_{var}_train.nc")['stream'][:]
				var_std = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningstd_{var}_train.nc')['stream'][:]
			else:
				var_mean = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningmean_{var}_train.nc')[var][:]
				var_std = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningstd_{var}_train.nc')[var][:]
			#breakpoint()
			var_out= np.empty_like(var_array)
			for year in range(years_n): 
				var_year = (var_array[year*122:(year+1)*122,:,:] - var_mean) / (3 * var_std)
				var_year = np.where(var_year < -1, -1, var_year) #remove outliers 
				var_year = np.where(var_year > 1, 1, var_year)
				var_out[year*122:(year+1)*122,:,:]=var_year
			outs.append(var_out)
 
		elif method == 'normalization':
			print("normalization method: out=(X - Xmin)/(Xmax-Xmin)")

			if var == 'stream250':
				var_max = Dataset(fr"{var}/{extent_name}_june2sept_7dayrunningmax_{var}_train.nc")['stream'][:]
				var_min = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningmin_{var}_train.nc')['stream'][:]
			else:
				var_max = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningmax_{var}_train.nc')[var][:]
				var_min = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningmin_{var}_train.nc')[var][:]
			#breakpoint()
			var_out= np.empty_like(var_array)
			for year in range(years_n): 
				var_year = (var_array[year*122:(year+1)*122,:,:] - var_min) / (var_max - var_min)
				var_out[year*122:(year+1)*122,:,:]=var_year
			outs.append(var_out)

		elif method == 'stand_norm_nans':
			print("stand + norma +nans method: out=(standardized_value - -3)/(3 - -3)")
			print("above 1 and below 0 are replaced with np.nan - outliers ")
			if var == 'stream250':
				var_mean = Dataset(fr"{var}/{extent_name}_june2sept_7dayrunningmean_{var}_train.nc")['stream'][:]
				var_std = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningstd_{var}_train.nc')['stream'][:]
			else:
				var_mean = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningmean_{var}_train.nc')[var][:]
				var_std = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningstd_{var}_train.nc')[var][:]
			#breakpoint()
			var_out= np.empty_like(var_array)
			for year in range(years_n): 
				var_year = (((var_array[year*122:(year+1)*122,:,:] - var_mean) / var_std ) + 3) / 6
				var_year = np.where(var_year < 0, np.NAN, var_year) #remove outliers 
				var_year = np.where(var_year > 1, np.NAN, var_year)
				var_out[year*122:(year+1)*122,:,:]=var_year

		elif method == 'stand_norm_01s':
			print("stand + norma +nans method: out=(standardized_value - -4)/(4 - -4)")
			print("above 1 and below 0 are replaced with 1 and 0 respecitvely - outliers ")
			if var == 'stream250':
				var_mean = Dataset(fr"{var}/{extent_name}_june2sept_7dayrunningmean_{var}_train.nc")['stream'][:]
				var_std = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningstd_{var}_train.nc')['stream'][:]
			else:
				var_mean = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningmean_{var}_train.nc')[var][:]
				var_std = Dataset(fr'{var}/{extent_name}_june2sept_7dayrunningstd_{var}_train.nc')[var][:]
			#breakpoint()
			var_out= np.empty_like(var_array)
			for year in range(years_n): 
				var_year = (((var_array[year*122:(year+1)*122,:,:] - var_mean) / var_std ) + 4) / 8
				var_year = np.where(var_year < 0, 0, var_year) #remove outliers 
				var_year = np.where(var_year > 1, 1, var_year)
				var_out[year*122:(year+1)*122,:,:]=var_year

			outs.append(var_out)


	os.chdir(curdir)
	return outs

def data_to_events(data_list:list, variables:list, event_len:int, ensemble_i:list, years_i:list, day0_i:list, years_ind:list):

	#PURPOSE: to create numpy array with [events, feature1, feature2, feature3]
	#INPUTS
	#data_list = list with data_array for each variable in variables 
	#variables = list with variables inside the data_arrat
	#event_len = number of days before and after day 0
	#ensemble_i = heatwave_list with ensemble member index
	#years_i = heatwave_list with years
	#day0_i = heatwave_list with starting day
	#years_ind = heatwave_list with index of ensemble_year
	#OUTPUT
	#events = list with for each event the three variable arrays
	
	print("Variable order is ", variables[:])
	stream_data = np.nanmean(data_list[0], axis=1)
	print("stream_data.shape is", stream_data.shape)
	psl_data = data_list[1]
	print("psl_data.shape is", psl_data.shape)
	events = []

	prev_event = []
	for i in range(len(years_i)):
		#select the correct data from the variables
		index_day = int(years_ind[i] * 122 + 30 - event_len + day0_i[i])
		cur_event = [ensemble_i[i], years_i[i], day0_i[i]] 
		if prev_event == cur_event:
			#print("duplicates")
			#print(prev_event, cur_event)
			continue
		else:
			prev_event = cur_event #update memory
			#index day in dataset is the year * days_in_year + 23 (june is 30 - 7days for start of event) + start_of_event_day 
			end = index_day + 2*event_len 
			f1 = stream_data[index_day:end,:,:]
			f2 = psl_data[index_day:end,:,:] 
			event = np.array([f1,f2])
			events.append(np.transpose(event)) #transpose such that shape becomes spatial, temporal, features
	return events

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	"""Returns a int64_list from a bool/enum/int/uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(event, parent, child, date):
	"""
	Creates a tf.train.Example message ready to be written to a file.
	"""
	# Create a dictionary mapping the feature name to the tf.train.Example-compatible
	# data type.
	feature = {
	  'features': _bytes_feature(tf.io.serialize_tensor(event)),
	  'parent':_int64_feature(parent),
	  'child': _int64_feature(int(child)), 
	  'year': _int64_feature(int(date[:4])),
	  'month': _int64_feature(int(date[5:7])),
	  'day': _int64_feature(int(date[8:])), 
	  # 'date': _bytes_feature(u"{date}".format(date=date).encode('utf-8')), #is er een tf.io optie voor string?
	}

	# Create a Features message using tf.train.Example.
	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto.SerializeToString()


def data_to_TFrecord(ensemble_number:int, path_heatwaves:str, variables:list, path_climate:str, 
					 years_n:int, extent_name:str, event_len:int, method:str):
	'''PURPOSE: to read in data from heatwaves and climate data, preform preprocessing steps, and write a
	TF record file
	
	INPUTS:
	ensemble_number:int = number of ensemble member
	path_heatwaves:str = path where heatwave data is stored
	variables:list of strings = variable names of climate data
	path_climate: string = of the path name where the climate data is stored
	years_n: integer =  the number of years in the ensemble
	extent: list of 4 integers = [lon_min,lon_max,lat_min,lat_max] is the spatial extent of the region of interest
	extent_name:str = name of extent
	event_len: int = number of days before and after event (e.g.7 means 7 days before and 7 days after day0)
	
	OUTPUTS:
	writes TF record file of all heatwave events 
	'''
	#READ IN HEATWAVE DATA
	ensemble_i, years_i, day0_i, years_ind, dates_i= loading_heatwave_data(ensemble_number, path_heatwaves)
	ens_year= [float(str(int(ensemble_i[i]))+str(years_i[i])) for i in range(len(ensemble_i))]  
	#READ AND PREPROCESS CLIMATE DATA
	list_of_data = preprocessing(variables, ensemble_number, path_climate, years_n, extent_name, method) 
	#CREATE HEATWAVE EVENTS
	events = data_to_events(list_of_data, variables, event_len, ensemble_i, years_i, day0_i, years_ind)
	

	#PROCESS EVENTS INTO TF RECORD FILES
	current_shard = 0
	img_in_current_shard = 0
	SAMPLES_PER_SHARD = len(events) #later zien of dit niet te groot is
	writer = tf.io.TFRecordWriter(f"files/TF_record_correctmask_ensemble{ensemble_number}_{extent_name}_{method}_noRSDS.tfrecord")
	for i, event in enumerate(events):
		if img_in_current_shard == SAMPLES_PER_SHARD:
			writer.close()
			#open new file
			current_shard += 1
			img_in_current_shard = 0
			writer = tf.io.TFRecordWriter(f"files/TF_record_correctmask_ensemble{ensemble_number}_{extent_name}_{method}_noRSDS.tfrecord")
		#process current sample and write to file
		tf_example = serialize_example(event, ensemble_number, ensemble_i[i], dates_i[i])
		#print(tf_example)
		writer.write(tf_example) #Serializetostring()
		img_in_current_shard += 1
	writer.close()



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-ens_member", type=int, help="ens_member", required=True)
	parser.add_argument("-path_heatwaves", type=str, help="path_heatwaves", required=True)
	parser.add_argument("-path_climate", type=str, help="path_climate", required=True)
	parser.add_argument("-years_n", type=int, help="years_n", required=True)
	parser.add_argument("-extent_name", type=str, help="extent_name", required=True)
	parser.add_argument("-event_len", type=int, help="event_len", required=True)
	parser.add_argument("-method", type=str, help="standardization_method", required=True)
	args = parser.parse_args()
	if args.ens_member < 101 or args.ens_member > 116:
		print("ensemble member not available!")
		sys.exit()
	data_to_TFrecord(args.ens_member, args.path_heatwaves, ["stream250", "psl"], args.path_climate, 
		args.years_n, args.extent_name, args.event_len, args.method)




