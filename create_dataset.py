import numpy as np
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import pandas as pd
import datetime


def strTodate(x):
	result=''
	for i in x.split('-'):
		for j in i:
			try:
				result+=str(int(j))
			except:
				continue
		result+='-'
	return result[:-1]


def load_enso_indices(instrument_data):
  """
  Reads in the txt data file to output a pandas Series of ENSO vals

  outputs
  -------

    pd.Series : monthly ENSO values starting from 1870-01-01
  """
  with open(instrument_data) as f:
    line = f.readline()
    enso_vals = []
    while line:
        yearly_enso_vals = map(float, line.split()[1:])
        enso_vals.extend(yearly_enso_vals)
        line = f.readline()

  enso_vals = pd.Series(enso_vals)
  enso_vals.index = pd.date_range('1870-01-01',freq='MS',
                                  periods=len(enso_vals))
  enso_vals.index = pd.to_datetime(enso_vals.index)
  return enso_vals

def assemble_basic_predictors_predictands(opt, train=False):
  """
  inputs
  ------

      start_date        str : the start date from which to extract sst
      end_date          str : the end date 
      lead_time         str : the number of months between each sst
                              value and the target Nino3.4 Index
      use_pca          bool : whether or not to apply principal components
                              analysis to the sst field
      n_components      int : the number of components to use for PCA

  outputs
  -------
      Returns a tuple of the predictors (np array of sst temperature anomalies) 
      and the predictands (np array the ENSO index at the specified lead time).

  """
  start_date = opt.startdate
  end_date = opt.enddate
  lead_time=opt.leadtime
  use_pca = opt.pca
  num_input_time_steps = opt.num_input_time_steps
  lat_slice=opt.lat_slice
  lon_slice=opt.lon_slice
  n_components = opt.n_components
  fname = opt.dataroot
  variable_name = opt.variable_name
  data_format = opt.data_format
  dataset = opt.dataset
  if opt.variable_name:
    variable_name  = opt.variable_name
  else:
    variable_name = {'observations' : 'sst',
                   'observations2': 't2m',
                   'CNRM'         : 'tas',
                   'MPI'          : 'tas'}[dataset]

  ds = xr.open_dataset(fname)
  sst = ds[variable_name].sel(time=slice(start_date, end_date))
  num_time_steps = sst.shape[0]
  sst = sst.values.reshape(num_time_steps, -1)
  sst[np.isnan(sst)] = 0
  if use_pca:
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca.fit(sst)
    X = pca.transform(sst)
  else:
    X = sst
  if train:
    sst1 = ds[variable_name].sel(time=slice(start_date, end_date))
    if lat_slice is not None:
      try:
          sst1=sst1.sel(lat=lat_slice)
      except:
          raise NotImplementedError("Implement slicing!")
    if lon_slice is not None:
      try:
          sst1=sst1.sel(lon=lon_slice)
      except:
          raise NotImplementedError("Implement slicing!")
    num_samples = sst1.shape[0]
    sst1 = np.stack([sst1.values[n-num_input_time_steps:n] for n in range(num_input_time_steps,
                                                              num_samples+1)])
    sst1[np.isnan(sst1)] = 0
    if data_format=='flatten':

      sst1 = sst1.reshape(num_samples, -1)

      if use_pca:
        pca = sklearn.decomposition.PCA(n_components=n_components)
        pca.fit(sst1)
        X1 = pca.transform(sst1)
      else:
        X1 = sst1
    else: # data_format=='spatial'
      X1 = sst1
    start_date_plus_lead1 = pd.to_datetime(start_date) + \
                        pd.DateOffset(months=lead_time+num_input_time_steps-1)
    end_date_plus_lead1 = pd.to_datetime(end_date) + \
                      pd.DateOffset(months=lead_time)
  
    X1 = X1.astype(np.float32)
    target_start_date_with_2_month1 = start_date_plus_lead1 - pd.DateOffset(months=2)
    subsetted_ds = ds[variable_name].sel(time=slice(target_start_date_with_2_month1,
                                                   end_date_plus_lead1))
    #Calculate the Nino3.4 index
    y = subsetted_ds.sel(lat=slice(5,-5), lon=slice(360-170,360-120)).mean(dim=('lat','lon'))

    y = pd.Series(y.values).rolling(window=3).mean()[2:].values
    y = y.astype(np.float32)
    #X = X[1:]
  else:
    start_date_plus_lead = pd.to_datetime(start_date) + \
                        pd.DateOffset(months=lead_time)
    end_date_plus_lead = pd.to_datetime(end_date) + \
                      pd.DateOffset(months=lead_time)
    if opt.compare_ground_truth:
      y = load_enso_indices(opt.instrument_data)[slice(start_date_plus_lead, 
                                end_date_plus_lead)]
    else:
      y = np.array([0]*X.shape[0])
  
##  ds = xr.open_dataset(opt.dataroot)
##  sst = ds['sst'].sel(time=slice(start_date, end_date))
##  num_time_steps = sst.shape[0]
##  
##  #sst is a 3D array: (time_steps, lat, lon)
##  #in this tutorial, we will not be using ML models that take
##  #advantage of the spatial nature of global temperature
##  #therefore, we reshape sst into a 2D array: (time_steps, lat*lon)
##  #(At each time step, there are lat*lon predictors)
##  sst = sst.values.reshape(num_time_steps, -1)
##  sst[np.isnan(sst)] = 0
##
##  #Use Principal Components Analysis, also called
##  #Empirical Orthogonal Functions, to reduce the
##  #dimensionality of the array
##  if use_pca:
##    pca = sklearn.decomposition.PCA(n_components=n_components)
##    pca.fit(sst)
##    X = pca.transform(sst)
##  else:
##    X = sst
##
##  start_date_plus_lead = pd.to_datetime(start_date) + \
##                        pd.DateOffset(months=lead_time)
##  end_date_plus_lead = pd.to_datetime(end_date) + \
##                      pd.DateOffset(months=lead_time)
##  y = load_enso_indices(opt.instrument_data)[slice(start_date_plus_lead, 
##                                end_date_plus_lead)]
##  #print(type(X))
##  #print(type(y))
##
##
##  ds.close()
  return X, y


def assemble_predictors_predictands(opt,train=False):
  """
  inputs
  ------

      start_date           str : the start date from which to extract sst
      end_date             str : the end date 
      lead_time            str : the number of months between each sst
                              value and the target Nino3.4 Index
      dataset              str : 'observations' 'CNRM' or 'MPI'
      data_format          str : 'spatial' or 'flatten'. 'spatial' preserves
                                  the lat/lon dimensions and returns an 
                                  array of shape (num_samples, num_input_time_steps,
                                  lat, lon).  'flatten' returns an array of shape
                                  (num_samples, num_input_time_steps*lat*lon)
      num_input_time_steps int : the number of time steps to use for each 
                                 predictor sample
      use_pca             bool : whether or not to apply principal components
                              analysis to the sst field
      n_components         int : the number of components to use for PCA
      lat_slice           slice: the slice of latitudes to use 
      lon_slice           slice: the slice of longitudes to use

  outputs
  -------
      Returns a tuple of the predictors (np array of sst temperature anomalies) 
      and the predictands (np array the ENSO index at the specified lead time).

  """
  if train:
    start_date = opt.startdate
    end_date = opt.enddate
  else:
    start_date = opt.test_start
    end_date = opt.test_end
  lead_time=opt.leadtime
  use_pca = opt.pca
  n_components = opt.n_components
  dataset = opt.dataset
  data_format = opt.data_format
  num_input_time_steps = opt.num_input_time_steps
  lat_slice = opt.lat_slice
  lon_slice = opt.lon_slice

  
  file_name = opt.dataroot
  if opt.variable_name:
    variable_name  = opt.variable_name
  else:
    variable_name = {'observations' : 'sst',
                   'observations2': 't2m',
                   'CNRM'         : 'tas',
                   'MPI'          : 'tas'}[dataset]

  ds = xr.open_dataset(file_name)
  sst = ds[variable_name].sel(time=slice(start_date, end_date))
  if lat_slice is not None:
    try:
        sst=sst.sel(lat=lat_slice)
    except:
        raise NotImplementedError("Implement slicing!")
  if lon_slice is not None:
    try:
        sst=sst.sel(lon=lon_slice)
    except:
        raise NotImplementedError("Implement slicing!")
  
  
  num_samples = sst.shape[0]
  #sst is a (num_samples, lat, lon) array
  #the line below converts it to (num_samples, num_input_time_steps, lat, lon)
  sst = np.stack([sst.values[n-num_input_time_steps:n] for n in range(num_input_time_steps,
                                                              num_samples+1)])
  #CHALLENGE: CAN YOU IMPLEMENT THE ABOVE LINE WITHOUT A FOR LOOP?
  num_samples = sst.shape[0]

  sst[np.isnan(sst)] = 0
  if data_format=='flatten':
    #sst is a 3D array: (time_steps, lat, lon)
    #in this tutorial, we will not be using ML models that take
    #advantage of the spatial nature of global temperature
    #therefore, we reshape sst into a 2D array: (time_steps, lat*lon)
    #(At each time step, there are lat*lon predictors)
    sst = sst.reshape(num_samples, -1)
    

    #Use Principal Components Analysis, also called
    #Empirical Orthogonal Functions, to reduce the
    #dimensionality of the array
    if use_pca:
      pca = sklearn.decomposition.PCA(n_components=n_components)
      pca.fit(sst)
      X = pca.transform(sst)
    else:
      X = sst
  else: # data_format=='spatial'
    X = sst

  start_date_plus_lead = pd.to_datetime(start_date) + \
                        pd.DateOffset(months=lead_time+num_input_time_steps-1)
  end_date_plus_lead = pd.to_datetime(end_date) + \
                      pd.DateOffset(months=lead_time)
  if dataset == 'observations' and opt.compare_ground_truth == True:
    y = load_enso_indices(opt.instrument_data)[slice(start_date_plus_lead, 
                                  end_date_plus_lead)]
  elif not opt.compare_ground_truth:
    y=np.array([0]*X.shape[0])
  else: #the data is from a GCM
    X = X.astype(np.float32)
    #The Nino3.4 Index is composed of three month rolling values
    #Therefore, when calculating the Nino3.4 Index in a GCM
    #we have to extract the two months prior to the first target start date
    target_start_date_with_2_month = start_date_plus_lead - pd.DateOffset(months=2)
    subsetted_ds = ds[variable_name].sel(time=slice(target_start_date_with_2_month,
                                                   end_date_plus_lead))
    #Calculate the Nino3.4 index
    y = subsetted_ds.sel(lat=slice(5,-5), lon=slice(360-170,360-120)).mean(dim=('lat','lon'))
    y = pd.Series(y.values).rolling(window=3).mean()[2:].values
    
    y = y.astype(np.float32)
  ds.close()
  return X.astype(np.float32), y.astype(np.float32)

class ENSODataset(Dataset):
    def __init__(self, predictors, predictands):
        self.predictors = predictors
        self.predictands = predictands
        assert self.predictors.shape[0] == self.predictands.shape[0], \
               "The number of predictors must equal the number of predictands!"

    def __len__(self):
        return self.predictors.shape[0]

    def __getitem__(self, idx):
        return self.predictors[idx], self.predictands[idx]

