import numpy as np 
from netCDF4 import Dataset
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import time
import keras

from utils_3d import *

##DEFINE AUTOENCODER #ARCHITECTURE

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, LEARNING_RATE=1e-3, filter_scaling=1, activation="relu", kernel_initializer="glorot_uniform", regularizer=tf.keras.regularizers.L2(0.1)):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    self.activation = activation
    if activation == "LeakyReLu":
      self.activation = keras.layers.LeakyReLU()
    self.kernel_initializer = kernel_initializer
    self.encoder = tf.keras.Sequential(
        [   
            tf.keras.layers.InputLayer(input_shape=(192, 64, 5, 2)),
            tf.keras.layers.Conv3D(filters=(32*filter_scaling), kernel_size=(3,3,3), strides=(2, 2, 1), activation=self.activation, 
               kernel_regularizer=regularizer, padding="same", kernel_initializer=self.kernel_initializer),
            tf.keras.layers.BatchNormalization(axis=4),
            tf.keras.layers.Conv3D(filters=(64*filter_scaling), kernel_size=(3,3,3), strides=(2,2,2), activation=self.activation, 
              kernel_regularizer=regularizer, padding="same", kernel_initializer=self.kernel_initializer),
            tf.keras.layers.BatchNormalization(axis=4),
            tf.keras.layers.Conv3D(filters=(96*filter_scaling), kernel_size=(3,3,3), strides=(3,2,3), activation=self.activation, 
              kernel_regularizer=regularizer, padding="same", kernel_initializer=self.kernel_initializer),
            tf.keras.layers.BatchNormalization(axis=4),
            tf.keras.layers.Conv3D(filters=(128*filter_scaling), kernel_size=(1,3,3), strides=(1,2,2), activation=self.activation, 
              kernel_regularizer=regularizer, padding="same", kernel_initializer=self.kernel_initializer),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=16*4*1*(128*filter_scaling), activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(16, 4,1,(128*filter_scaling))),
            tf.keras.layers.Conv3DTranspose(filters=(96*filter_scaling), kernel_size=(1,3,1), strides=(1,2,1), padding='same', activation=self.activation, 
              kernel_regularizer=regularizer, kernel_initializer=self.kernel_initializer),
            tf.keras.layers.BatchNormalization(axis=4),
            tf.keras.layers.Conv3DTranspose(filters=(64*filter_scaling), kernel_size=(3,3,3), strides=(3,2,3), padding='same', activation=self.activation, 
              kernel_regularizer=regularizer, kernel_initializer=self.kernel_initializer),
            tf.keras.layers.BatchNormalization(axis=4),
            tf.keras.layers.Conv3DTranspose(filters=(32*filter_scaling), kernel_size=(3,3,3), strides=(2,2,1), padding='valid', activation=self.activation, 
              kernel_regularizer=regularizer, kernel_initializer=self.kernel_initializer),
            tf.keras.layers.BatchNormalization(axis=4),
            tf.keras.layers.Cropping3D(cropping=((0, 1), (0, 1), (0, 0))),
            tf.keras.layers.Conv3DTranspose(filters=(32*filter_scaling), kernel_size=(3,3,3), strides=(2,2,1), padding='same', activation=self.activation,
              kernel_regularizer=regularizer, kernel_initializer=self.kernel_initializer),
            tf.keras.layers.BatchNormalization(axis=4),
            tf.keras.layers.Conv3DTranspose(filters=2, kernel_size=(3,3,3), strides=(1,1,1), padding='same', kernel_regularizer=regularizer),
        
        ]
    )


  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

  def call(self, x):
    mean, logvar = self.encode(x)

    batch = tf.shape(mean)[0]
    dim   = tf.shape(mean)[1]
    eps = tf.random.normal(shape=(batch, dim))
    z = eps * tf.exp(logvar * .5) + mean

    return self.decode(z)
 
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x, beta=1, LOSS="MSE", sparse=False, WEIGHT_DECAY_2=0.05):
  #to compute the loss
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  if LOSS == 'MSE':
    MSE = tf.keras.losses.mean_squared_error(x, x_logit)
    MSE_per_batch = tf.reduce_sum(MSE, axis=[1,2,3])
    logpx_z = tf.reduce_mean(MSE_per_batch)
  elif LOSS == "LOG_COSH":
    LOG_COSH = tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.SUM)
    logpx_z = LOG_COSH(x, x_logit)
  elif LOSS == "CROSSENTROPY":
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = tf.reduce_sum(cross_ent, axis=[1, 2, 3, 4]) #added ,4
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  kl_loss = -beta * (logpz - logqz_x)
  reg = tf.reduce_sum(model.losses)
  if not sparse:
    encoded_l1_loss = 0
  if sparse:
    encoded_l1_loss = WEIGHT_DECAY_2 * tf.reduce_sum(abs(mean))
  loss = tf.reduce_mean(logpx_z + kl_loss + encoded_l1_loss + reg)
  return loss, logpx_z, kl_loss, encoded_l1_loss, reg

def compute_loss_ae(model, x, beta=0, LOSS="MSE", sparse=False, WEIGHT_DECAY_2=0.05):
  mean, logvar = model.encode(x)
  x_logit = model.decode(mean)
  if LOSS == 'MSE':
    MSE = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)    #reduction=tf.keras.losses.Reduction.SUM
    logpx_z = MSE(x, x_logit)
    # logpx_z = - tf.keras.losses.mean_squared_error(x, x_logit)
  elif LOSS == "LOG_COSH":
    LOG_COSH = tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.SUM)
    logpx_z = LOG_COSH(x, x_logit)
  elif LOSS == "CROSSENTROPY":
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = tf.reduce_sum(cross_ent, axis=[1, 2, 3, 4]) #added ,4
  if not sparse:
    loss = tf.reduce_mean(logpx_z + (beta * (logpz - logqz_x)))
  if sparse:
    encoded_l1_loss = WEIGHT_DECAY_2 * tf.reduce_sum(abs(mean))
    reg = tf.reduce_sum(model.losses)
    loss = tf.reduce_mean(logpx_z + encoded_l1_loss + reg)
  return loss, logpx_z, encoded_l1_loss, reg


@tf.function
def train_step(model, x, beta=1, LOSS='MSE', ae=False, sparse=False, WEIGHT_DECAY_2=0.05):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    if not ae:
      #print("ae is false")
      loss, logpx_z, vae_reg, encoded_l1_loss, reg = compute_loss(model, x, beta=beta, LOSS=LOSS, sparse=sparse, WEIGHT_DECAY_2=WEIGHT_DECAY_2)
    if ae:
      loss, logpx_z, encoded_l1_loss, reg = compute_loss_ae(model, x, beta=beta, LOSS=LOSS, sparse=sparse, WEIGHT_DECAY_2=WEIGHT_DECAY_2)
      vae_reg = 0
  gradients = tape.gradient(loss, model.trainable_variables)
  model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss, logpx_z, vae_reg, encoded_l1_loss, reg
