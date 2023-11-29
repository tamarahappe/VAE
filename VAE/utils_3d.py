import numpy as np 
from netCDF4 import Dataset
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import time


def load_tfrecords(ensembles, norm_method='normalization'):
#     os.chdir(path)
    filenames = []
    if norm_method == 'standardized':
      norm_method = "0"
    for ens in ensembles:
        #ADJUST this filename accordingly
        filenames.append(f"/home/thappe/tf_records/files/TF_record_correctmask_ensemble{ens}_NAext2_{norm_method}_noRSDS.tfrecord")
    data = tf.data.TFRecordDataset(filenames)
    return data

#Define features saved in tf_record
feature_description = {
'features': tf.io.FixedLenFeature([], tf.string),
'parent': tf.io.FixedLenFeature([], tf.int64), 
'child': tf.io.FixedLenFeature([], tf.int64),
'year': tf.io.FixedLenFeature([], tf.int64),
'month': tf.io.FixedLenFeature([], tf.int64),
'day': tf.io.FixedLenFeature([], tf.int64),
}


#Create functions to read tf_records
def _parse_function(example):
    """
    Takes in example event from the raw data (tensorflow dataset), and returns the original values
    
    if test==True, the example is returned with all information attached. Otherwise just the features are returned
    for the use of model training.
    """
    parsed_example = tf.io.parse_single_example(example, feature_description)

    features = parsed_example['features']
    features = tf.io.parse_tensor(features, tf.float32)
    return features[:,:,7:12,:], features[:,:,7:12,:] #to return only 5 days


#Create functions to read tf_records
def _parse_function_augment(example):
    """
    Takes in example event from the raw data (tensorflow dataset), and returns the original values
    
    if test==True, the example is returned with all information attached. Otherwise just the features are returned
    for the use of model training.
    """
    parsed_example = tf.io.parse_single_example(example, feature_description)

    features = parsed_example['features']
    features = tf.io.parse_tensor(features, tf.float32)
    return features[:,:,7:12,:]
    

def _parse_function_full(example):
    """
    Takes in example event from the raw data (tensorflow dataset), and returns the original values
    
    if test==True, the example is returned with all information attached. Otherwise just the features are returned
    for the use of model training.
    """
    parsed_example = tf.io.parse_single_example(example, feature_description)
    
    parent = parsed_example['parent']
    child = parsed_example['child']

    features = parsed_example['features']
    features = tf.io.parse_tensor(features, tf.float32)

    year = parsed_example['year']
    month = parsed_example['month']
    day = parsed_example['day']

    return parent, child, features[:,:,7:12,:], year, month, day


def create_pair(sample):
  return sample, sample

def augment(sample):
  """Perform random augmentations on a sample with dimensions [192, 64, 5, 2]"""
  #spatial flips
  if  tf.random.uniform([]) < 0.5:
    sample = tf.reverse(sample, axis=[0])
  if  tf.random.uniform([]) < 0.5:
    sample = tf.reverse(sample, axis=[1])

  #brightness
  #sample = tf.image.random_brightness(sample, max_delta=0.25)

  #contrast
  sample = tf.image.random_contrast(sample, lower=0.9, upper=1.1)

  #random crop and resize
  #in 20% of cases, the original image is used for training
  #in 80% of cases, a crop factor of 0.9375 is applied in each direction for an area of approx. 88%
  if  tf.random.uniform([]) > 0.2:
    sample = tf.image.random_crop(sample, size=[180, 60, 5, 2])
    #swap columns (use batch size slot to temporarily store time dimension)
    sample = tf.transpose(sample, perm=[2, 0, 1, 3])
    sample = tf.image.resize(sample, size=[192, 64])
    #swap columns back
    sample = tf.transpose(sample, perm=[1, 2, 0, 3])

  #ensure values are not out of bounds
  sample = tf.clip_by_value(sample, -1, 1)

  return sample, sample



lons = np.array([-74.53125 , -73.828125, -73.125   , -72.421875,
                   -71.71875 , -71.015625, -70.3125  , -69.609375,
                   -68.90625 , -68.203125, -67.5     , -66.796875,
                   -66.09375 , -65.390625, -64.6875  , -63.984375,
                   -63.28125 , -62.578125, -61.875   , -61.171875,
                   -60.46875 , -59.765625, -59.0625  , -58.359375,
                   -57.65625 , -56.953125, -56.25    , -55.546875,
                   -54.84375 , -54.140625, -53.4375  , -52.734375,
                   -52.03125 , -51.328125, -50.625   , -49.921875,
                   -49.21875 , -48.515625, -47.8125  , -47.109375,
                   -46.40625 , -45.703125, -45.      , -44.296875,
                   -43.59375 , -42.890625, -42.1875  , -41.484375,
                   -40.78125 , -40.078125, -39.375   , -38.671875,
                   -37.96875 , -37.265625, -36.5625  , -35.859375,
                   -35.15625 , -34.453125, -33.75    , -33.046875,
                   -32.34375 , -31.640625, -30.9375  , -30.234375,
                   -29.53125 , -28.828125, -28.125   , -27.421875,
                   -26.71875 , -26.015625, -25.3125  , -24.609375,
                   -23.90625 , -23.203125, -22.5     , -21.796875,
                   -21.09375 , -20.390625, -19.6875  , -18.984375,
                   -18.28125 , -17.578125, -16.875   , -16.171875,
                   -15.46875 , -14.765625, -14.0625  , -13.359375,
                   -12.65625 , -11.953125, -11.25    , -10.546875,
                    -9.84375 ,  -9.140625,  -8.4375  ,  -7.734375,
                    -7.03125 ,  -6.328125,  -5.625   ,  -4.921875,
                    -4.21875 ,  -3.515625,  -2.8125  ,  -2.109375,
                    -1.40625 ,  -0.703125,   0.      ,   0.703125,
                     1.40625 ,   2.109375,   2.8125  ,   3.515625,
                     4.21875 ,   4.921875,   5.625   ,   6.328125,
                     7.03125 ,   7.734375,   8.4375  ,   9.140625,
                     9.84375 ,  10.546875,  11.25    ,  11.953125,
                    12.65625 ,  13.359375,  14.0625  ,  14.765625,
                    15.46875 ,  16.171875,  16.875   ,  17.578125,
                    18.28125 ,  18.984375,  19.6875  ,  20.390625,
                    21.09375 ,  21.796875,  22.5     ,  23.203125,
                    23.90625 ,  24.609375,  25.3125  ,  26.015625,
                    26.71875 ,  27.421875,  28.125   ,  28.828125,
                    29.53125 ,  30.234375,  30.9375  ,  31.640625,
                    32.34375 ,  33.046875,  33.75    ,  34.453125,
                    35.15625 ,  35.859375,  36.5625  ,  37.265625,
                    37.96875 ,  38.671875,  39.375   ,  40.078125,
                    40.78125 ,  41.484375,  42.1875  ,  42.890625,
                    43.59375 ,  44.296875,  45.      ,  45.703125,
                    46.40625 ,  47.109375,  47.8125  ,  48.515625,
                    49.21875 ,  49.921875,  50.625   ,  51.328125,
                    52.03125 ,  52.734375,  53.4375  ,  54.140625,
                    54.84375 ,  55.546875,  56.25    ,  56.953125,
                    57.65625 ,  58.359375,  59.0625  ,  59.765625])
lats = np.array([30.5262516 , 31.22800418, 31.92975673, 32.63150925,
                   33.33326174, 34.0350142 , 34.73676663, 35.43851902,
                   36.14027138, 36.8420237 , 37.54377599, 38.24552823,
                   38.94728044, 39.6490326 , 40.35078471, 41.05253678,
                   41.75428879, 42.45604076, 43.15779267, 43.85954452,
                   44.56129631, 45.26304804, 45.9647997 , 46.66655129,
                   47.3683028 , 48.07005424, 48.7718056 , 49.47355688,
                   50.17530806, 50.87705915, 51.57881013, 52.28056101,
                   52.98231178, 53.68406242, 54.38581295, 55.08756333,
                   55.78931357, 56.49106366, 57.19281359, 57.89456335,
                   58.59631292, 59.2980623 , 59.99981146, 60.7015604 ,
                   61.40330909, 62.10505753, 62.80680568, 63.50855352,
                   64.21030104, 64.9120482 , 65.61379497, 66.31554132,
                   67.01728721, 67.71903259, 68.42077741, 69.12252163,
                   69.82426517, 70.52600796, 71.22774993, 71.92949096,
                   72.63123095, 73.33296977, 74.03470726, 74.73644324]) 

def visualize_results(data_testbatch, model, epoch, DIR, show=True, returnfig=False, norm_method="normalization"):  
    import warnings
    warnings.filterwarnings( "ignore" )
    
    to_predict = np.zeros((1, 192, 64, 5, 2))
    
    for element in data_testbatch.as_numpy_iterator(): #to convert to numpy elements
        array = np.transpose(element[2])
        to_predict[0] = element[2]
        stream_in = array[0, 0, :, :] 
        psl_in = array[1, 0, :, :]
        date = str(element[3]) + "_" + str(element[4]) + "_" + str(element[5]) 
        ens = str(element[0]) + "_" + str(element[1])
    
    #predict output
    mean, logvar = model.encode(to_predict)
    z = model.reparameterize(mean, logvar)
    predicted = model.decode(z, apply_sigmoid=False)
    
    #make sure predicted is in the right format (e.g. array, use np.transpose)
    stream_pred = np.transpose(predicted[0,:,:,0, 0])
    psl_pred = np.transpose(predicted[0,:,:,0, 1])


    to_plot = ["stream_in", "stream_pred","psl_in", "psl_pred" ]    
    plotdict = {"stream_in":[stream_in, 221, "stream function at 250hpa - input"], "stream_pred":[stream_pred, 222,"stream function at 250hpa - predicted"],
               "psl_in":[psl_in, 223, "sea level pressure - input"], "psl_pred":[psl_pred, 224, "sea level pressure - predicted"]}
        
    cmap='seismic'
    fig, axes = plt.subplots(2,2, figsize=(20,12))
 
    if norm_method == "standardized":
      vmin, vmax = -3, 3
    elif norm_method == "standardization_cut":
      vmin, vmax = -1,1
    else:
      vmin, vmax = 0, 1

    for i, var in enumerate(to_plot):
        ax = plt.subplot(plotdict[var][1], projection=ccrs.PlateCarree())
        cs = ax.pcolormesh(lons, lats,plotdict[var][0] ,
                 transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax) #min max aanpassen voor norm/stnd method
        ax.coastlines()
        ax.set_title(f"Standardized {plotdict[var][2]}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        fig.colorbar(cs, ax=ax, fraction=0.018, pad=0.02)

      
    fig.suptitle(f"Autoencoder results - example heatwave day0 - epoch {str(epoch).zfill(4)}", fontsize=20)
    plt.savefig(f"{DIR}/visualisation_{str(epoch).zfill(4)}.png", dpi=100) 
    #plt.savefig("test.png", dpi=100)
    if show == True:  
        plt.show()

    if returnfig == True:
      return fig
    plt.close()


def visualize_loss(elbos, epochs, DIR, epoch, show=True):
    
    plt.plot(elbos, label="elbo loss", linewidth=4)
    plt.xlim([0, epochs])
    plt.ylim([np.min(elbos), -20000])
    plt.title(f"elbo loss epoch {epoch}")
    plt.savefig(f"{DIR}/{epoch}_elbo_loss.png", dpi=100)
    if show == True:
      plt.show()
    plt.close()

