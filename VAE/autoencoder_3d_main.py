import numpy as np 
import cartopy.crs as ccrs
from netCDF4 import Dataset
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import wandb

from utils_3d import _parse_function_full, _parse_function, _parse_function_augment, augment, create_pair, visualize_results, visualize_loss
from autoencoder_3d_model import *


def MAIN(
  SHUFFLE_SIZE=10000, #
  BATCH_SIZE = 4, 
  BATCH_SIZE_VAL = 256, # 
  epochs = 1000000, # 
  beta = 1,
  EXP_NAME = "exp0",
  DIR_CHECKPOINT = f"/home/thappe/VAE3D/autoencoder_results/exp0",
  DIR_figs = f"/home/thappe/VAE3D/autoencoder_results/exp0/figs",
  LEARNING_RATE = 1e-3,
  STEPS_PER_EPOCH = 12164, 
  latent_dim =100,
  PROJECT = "test-project",
  TAGS = [],
  LOGGING_FIGURES=False,
  norm_method = 'normalization',
  LOSS="MSE",
  LR_DECAY=False,
  LR_DECAY_RATE=1,
  filter_scaling=1,
  activation="relu",
  kernel_initializer="glorot_uniform",
  ae=False,
  WEIGHT_DECAY=0.1,
  sparse=False,
  WEIGHT_DECAY_2=0.05,
  AUGMENT=True,
  PRETRAINED_MODEL=None,
  ):
  print("DIR_CHECKPOINT is: ", DIR_CHECKPOINT)
  print("DIR_figs is: ", DIR_figs)

  if not os.path.isdir(DIR_CHECKPOINT): #if dir doesn't exist:
    os.makedirs(DIR_CHECKPOINT)

  if not os.path.isdir(DIR_figs):
    os.makedirs(DIR_figs)

  #converting values into floats
  LEARNING_RATE = float(LEARNING_RATE)
  beta = float(beta)

  STEPS_PER_EPOCH = STEPS_PER_EPOCH // BATCH_SIZE #to make sure we have covered all samples ,

  #LOAD data from records
  data_train_raw = load_tfrecords(np.arange(101,115,1), norm_method)
  data_val_raw = load_tfrecords(np.arange(115,117,1), norm_method)

  #Get a sample to use for visualisation during training
  data_testbatch = data_val_raw.map(_parse_function_full).take(1) #selecting only one for visualisation

  if AUGMENT: 
    #if we want to augment our data
    data_train = data_train_raw.map(_parse_function_augment)
    data_val = data_val_raw.map(_parse_function_augment) 

    train_batches = (data_train
                     .cache() #zodat de files in mem worden geplaats en niet elke keer opnieuw worden ingelezen
                     .shuffle(SHUFFLE_SIZE) #zodat het randomized wordt
                     .map(augment, num_parallel_calls=tf.data.AUTOTUNE) #data augmentation on train only
                     .batch(BATCH_SIZE) #how many samples per training step (dus zo hoog mogelijk zonder error)
                     .prefetch(buffer_size=tf.data.AUTOTUNE) #how many batches you prepare to have ready
                    )

    val_batches = (data_val
                   .cache()
                   .map(create_pair, num_parallel_calls=tf.data.AUTOTUNE) #create pair from the single samples from parse
                   .batch(BATCH_SIZE_VAL)
                   )
  if not AUGMENT:
    data_train = data_train_raw.map(_parse_function)
    data_val = data_val_raw.map(_parse_function) 

    train_batches = (data_train
                     .cache() #zodat de files in mem worden geplaats en niet elke keer opnieuw worden ingelezen
                     .shuffle(SHUFFLE_SIZE) #zodat het randomized wordt
                     .batch(BATCH_SIZE) #how many samples per training step (dus zo hoog mogelijk zonder error)
                     .prefetch(buffer_size=tf.data.AUTOTUNE) #how many batches you prepare to have ready
                    )

    val_batches = (data_val
                   .cache()
                   .batch(BATCH_SIZE_VAL)
                   )

  num_examples_to_generate = 16

  # keeping the random vector constant for generation (prediction) so
  # it will be easier to see the improvement.
  random_vector_for_generation = tf.random.normal(
      shape=[num_examples_to_generate, latent_dim])

  regularizer=tf.keras.regularizers.L2(float(WEIGHT_DECAY))
  model = CVAE(latent_dim, LEARNING_RATE, int(filter_scaling), activation=activation, kernel_initializer=kernel_initializer, regularizer=regularizer)

  if PRETRAINED_MODEL != None:
    #transfer learning:
    #load weights from pre-trained model
    model.build(input_shape=(None, 192, 64, 5, 2))
    model.load_weights(PRETRAINED_MODEL, skip_mismatch=True, by_name=True)

    

  #initialize metrics to collect losses during training/validation loops
  metric_total_loss_train = tf.keras.metrics.Mean()
  metric_MSE_train        = tf.keras.metrics.Mean()
  metric_reg_train        = tf.keras.metrics.Mean()
  metric_vae_reg_train    = tf.keras.metrics.Mean()
  metric_encoded_train    = tf.keras.metrics.Mean()
  metric_total_loss_val   = tf.keras.metrics.Mean()
  metric_MSE_val          = tf.keras.metrics.Mean()
  metric_reg_val          = tf.keras.metrics.Mean()
  metric_vae_reg_val      = tf.keras.metrics.Mean()
  metric_encoded_val      = tf.keras.metrics.Mean()

  for epoch in range(1, epochs + 1):
    #TRAINING
      start_time = time.time()

      for train_x in train_batches: 
          loss_train, MSE_train, vae_reg_train, encoded_loss_train, reg_train = train_step(model, train_x[0], beta, LOSS, ae, sparse=sparse, WEIGHT_DECAY_2=WEIGHT_DECAY_2)
          
          #update losses 
          metric_total_loss_train.update_state(loss_train)
          metric_MSE_train.update_state(MSE_train)
          metric_reg_train.update_state(reg_train)
          metric_vae_reg_train.update_state(vae_reg_train)
          metric_encoded_train.update_state(encoded_loss_train)

      loss_train_epoch    = metric_total_loss_train.result()
      MSE_train_epoch     = metric_MSE_train.result()
      reg_train_epoch     = metric_reg_train.result()
      vae_reg_train_epoch = metric_vae_reg_train.result()
      encoded_train_epoch = metric_encoded_train.result()
      end_time = time.time()

      start_time_val = time.time()

      for test_x in val_batches:
          if not ae:
            loss_val, MSE_val, vae_reg_val, encoded_loss_val, reg_val = compute_loss(model, test_x[0], beta=beta, LOSS=LOSS, sparse=sparse, WEIGHT_DECAY_2=WEIGHT_DECAY_2)
          if ae:
            vae_reg_val = 0
            loss_val, MSE_val, encoded_loss_val, reg_val = compute_loss_ae(model, test_x[0], beta=beta, LOSS=LOSS, sparse=sparse, WEIGHT_DECAY_2=WEIGHT_DECAY_2)

          metric_total_loss_val.update_state(loss_val)
          metric_MSE_val.update_state(MSE_val)
          metric_reg_val.update_state(reg_val)
          metric_vae_reg_val.update_state(vae_reg_val)
          metric_encoded_val.update_state(encoded_loss_val)

      loss_val_epoch    = metric_total_loss_val.result()
      MSE_val_epoch     = metric_MSE_val.result()
      reg_val_epoch     = metric_reg_val.result()
      vae_reg_val_epoch = metric_vae_reg_val.result()
      encoded_val_epoch = metric_encoded_val.result()
      end_time_val = time.time()

      learning_rate_current = model.optimizer.lr 
      if LR_DECAY:
        model.optimizer.lr.assign((1 / (1 + float(LR_DECAY_RATE) * epoch)) * learning_rate_current)


      #LOGGING INFORMATION
      print(f'Epoch: {epoch}, loss training set: {loss_train_epoch}, loss validation set: {loss_val_epoch}, time elapsed for current epoch, training: {end_time - start_time}; validation: {end_time_val - start_time_val}')     
      fig = None
      if (epoch % 10) == 0:
        if LOGGING_FIGURES == True:
          fig = visualize_results(data_testbatch, model, epoch, DIR_figs, show=False, returnfig=True, norm_method=norm_method)
          fig = wandb.Image(fig)
        else:
          visualize_results(data_testbatch, model, epoch, DIR_figs, show=False, returnfig=False, norm_method=norm_method)
          fig = None

      wandb.log({"epoch": epoch, "elbo_loss_train": np.mean(loss_train_epoch), "elbo_loss_val": np.mean(loss_val_epoch),
                 "MSE_val": np.mean(MSE_val_epoch), "MSE_train": np.mean(MSE_train_epoch),
                 "encoded_loss_val": encoded_val_epoch, "encoded_loss_train": encoded_train_epoch,
                 "reg_val": reg_val_epoch, "reg_train": reg_train_epoch,
                 "kl_loss_val": vae_reg_val_epoch, "kl_loss_train": vae_reg_train_epoch, 
                 "Learning_rate_ADAM": learning_rate_current, "fig": fig })
        
      if (epoch - 1) % 10 == 0 :
        model.save_weights(f"{DIR_CHECKPOINT}/VAE3D_{EXP_NAME}_{str(epoch).zfill(4)}.h5")

      #reset metrics before start of next epoch
      metric_total_loss_train.reset_states()
      metric_MSE_train.reset_states()
      metric_reg_train.reset_states()
      metric_vae_reg_train.reset_states()
      metric_encoded_train.reset_states()
      metric_total_loss_val.reset_states()
      metric_MSE_val.reset_states()
      metric_reg_val.reset_states()
      metric_vae_reg_val.reset_states()
      metric_encoded_val.reset_states()


  model.save_weights(f"{DIR_CHECKPOINT}/VAE3D_{EXP_NAME}_final.h5")
  visualize_results(data_testbatch, model, epoch, DIR_figs, show=False, returnfig=False, norm_method=norm_method)


wandb.finish()

if __name__ == '__main__':

    import os

    os.chdir("/home/thappe/VAE")

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--SHUFFLE_SIZE", type=int, default=10000, help="shuffle size for data batches")
    parser.add_argument("--BATCH_SIZE", type=int, default=4, help="size of batches")
    parser.add_argument("--BATCH_SIZE_VAL", type=int, default=265, help="size of batches for validation")
    parser.add_argument("--epochs", type=int, default=1000000, help="amount of epochs to train")
    parser.add_argument("--beta", default=1, help="beta param for weighting KLD vs reconstruction loss")
    parser.add_argument("--EXP_NAME", type=str, default="exp0", help="name of experiment, should be output dir existing")
    parser.add_argument("--DIR_CHECKPOINT", type=str, default="/home/thappe/autoencoder_results/",  help="directory where to save the model")
    parser.add_argument("--DIR_figs", type=str, default="/home/thappe/autoencoder_results/", help="directory where to save the figures")
    parser.add_argument("--LEARNING_RATE", default=1e-4, help="learning rate for the training steps") #type=int,
    parser.add_argument("--STEPS_PER_EPOCH", type=int, default=12164, help="amount of steps per epochs")
    parser.add_argument("--latent_dim", type=int, default=100, help="size of latent dimension of the model")
    parser.add_argument("--PROJECT", type=str, default="3D_noRSDS_5days", help="project space of WANDB where to log")
    parser.add_argument("--TAGS", default=[], help="Tags for WANDB")
    parser.add_argument("--LOGGING_FIGURES", type=bool, default=False, help="whether to log figures for WANDB or not") #this is not working yet
    parser.add_argument("--norm_method", type=str, default='normalization', help="type of normalization method of input data. default= normalization, other options = stand_norm_01s")
    parser.add_argument("--LOSS", type=str, default="MSE", help=" loss function = MSE, other options: LOG_COSH and CROSSENTROPY")
    parser.add_argument("--LR_DECAY", type=bool, default=False, help="Whether learning rate is decaying with epochs following lr_n=(1/(1+decay_rate*epochNumber))*LR")
    parser.add_argument("--LR_DECAY_RATE", default=1, help="decay rate in lr_n=(1/(1+decay_rate*epochNumber))*LR")    
    parser.add_argument("--filter_scaling", default=1, help="scaling factor to increase model complexity through filter size")
    parser.add_argument("--activation", default="relu", help="activation layer in the CVAE, default is relu")
    parser.add_argument("--kernel_initializer", default="glorot_uniform", help="type of kernal kernel_initializer for 3Dconv layers, default is glorot_uniform")
    parser.add_argument("--ae", default=False, help="To turn of the variational part in the loss computation set to True (so no resampling and reparameterization)")
    parser.add_argument("--WEIGHT_DECAY", default=0.1, help="weight decay for kernal regularizer, which is set to l2_regularizer")
    parser.add_argument("--sparse", default=False, help="Whether the autoencoder is sparse, if true; it will force the features in latent space to be more independent. if True; you can change WEIGHT_DECAY_2 argument")
    parser.add_argument("--WEIGHT_DECAY_2", default=0.05, help="weight decay/importance for sparsity of autoencoder")
    parser.add_argument("--AUGMENT", default=True, help="Whether to apply data augmentation to train set")
    parser.add_argument("--PRETRAINED_MODEL", default=None, help="file to pre-trained for tranfer learning with model weights")

    args = parser.parse_args()


    config = {
      "rsds": "no_rsds",
      "learning_rate": args.LEARNING_RATE,
      "epochs": args.epochs,
      "beta": float(args.beta),
      "batch_size": args.BATCH_SIZE,
      "batch_size_val": args.BATCH_SIZE_VAL,
      "latent_dims": args.latent_dim,
      "exp_name": args.EXP_NAME,
      "norm_method": args.norm_method,
      "filter_scaling": args.filter_scaling,
      "LR_DECAY_RATE": args.LR_DECAY_RATE,
      "activation":args.activation,
      "loss_function":args.LOSS,
      "kernel_initialzer":args.kernel_initializer,
      "ae":args.ae,
      "weight_decay_regulizer": args.WEIGHT_DECAY,
      "sparse":args.sparse,
      "WEIGHT_DECAY_sparcity":args.WEIGHT_DECAY_2,
      "data_augmentation": args.AUGMENT,
      "PRETRAINED_MODEL": args.PRETRAINED_MODEL
    }

    wandb.init(config=config, project=args.PROJECT, entity="heatwave_clusters", tags=args.TAGS)


    MAIN(**vars(args))


















