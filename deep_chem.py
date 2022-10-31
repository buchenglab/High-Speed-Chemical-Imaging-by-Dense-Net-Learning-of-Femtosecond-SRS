#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 20:12:00 2020
@author: jingzhang
"""
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import scipy.misc
import tensorflow as tf 
from tensorflow.keras.models import *
from tensorflow.keras.layers import (concatenate, Input, Conv2D, MaxPooling2D, Dropout,
    AveragePooling2D, UpSampling2D,BatchNormalization, ReLU, Conv2DTranspose, Activation, ZeroPadding2D)
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras


def deep_chem(para,input_size=(256,256,1),pretrained_weights = None):
    global kerel_regularizer_, kernel_size
    if para['kernel_regular']['class'] == 'l1':
        kerel_regularizer_ = l1(para['kernel_regular']['rate'])#l1(0.000001)
    if para['kernel_regular']['class'] == 'l2':
        kerel_regularizer_ = l2(para['kernel_regular']['rate'])
    n_block = para['n_block']
    filter_dim = para['filter_dim']
    neck_feature_dim = para['neck_feature_dim']
    down_dense_layers = para['down_dense_layers']
    up_layers = para['up_layers']
    alpha4loss = para['alpha4loss']
    inputs = Input(input_size)
    conv_1 = Conv2D(8,1,padding = 'same', name='conv1',kernel_regularizer = kerel_regularizer_)(inputs)
    ## downsampling
    module = []
    for nn in range(n_block):#n_block = 5
        dense_1 = DenseBlock(conv_1, 2, down_dense_layers[nn], nn)
        conc_1 = concatenate([conv_1,dense_1])
        module.append(conc_1) ## for skip connection
        TD_1 = TransDown(conc_1, down_dense_layers[nn], nn)
        conv_1 = TD_1
    ## bottleneck 
    bottle_layer = DenseBlock(conv_1, 2, neck_feature_dim, 6)## 4x4 after n_block = 5
    for mm in range(n_block):
        if mm == 0:
            up_1 = TransUp(bottle_layer, up_layers[0], mm, kernel_dim=(3,3))
        else:
            up_1 = TransUp(conc_1, up_layers[mm], mm, kernel_dim=(3,3))
        conc_1 = concatenate([up_1, module[-(mm+1)]])  

    conv_last = Conv2D(8, (3,3), padding = 'same',name = 'conv_last',kernel_regularizer = kerel_regularizer_)(conc_1)
    output_layer = Conv2D(1, (1,1), activation = 'sigmoid', padding = 'same')(conv_last)
    model = Model(inputs = inputs, outputs = output_layer, name = 'deep_chem')
    model.compile(optimizer = Adam(lr = 1e-4), loss =  crossent_recall(alpha4loss), metrics = ['accuracy',precision,recall])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = soft_dice_loss, metrics = ['accuracy'])
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model

def crossent_recall(alpha):
    def loss_(y_true,y_pred):
        crossentropy_ = binary_crossentropy(y_true,y_pred)
        recall_ = recall(y_true,y_pred)
        return alpha * crossentropy_ + (1 - alpha) * (1 - recall_)
    return loss_
def precision(y_true, y_pred): #taken from old keras source code
     true_positives = keras.sum(keras.round(keras.clip(y_true * y_pred, 0, 1)))
     predicted_positives = keras.sum(keras.round(keras.clip(y_pred, 0, 1)))
     precision = true_positives / (predicted_positives + keras.epsilon())
     return precision
def recall(y_true, y_pred): #taken from old keras source code
     true_positives = keras.sum(keras.round(keras.clip(y_true * y_pred, 0, 1)))
     possible_positives = keras.sum(keras.round(keras.clip(y_true, 0, 1)))
     recall = true_positives / (possible_positives + keras.epsilon())
     return recall
def tversky_loss(beta):
  def loss(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
    return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)
  return loss

def TransUp(input_layer, filter_dim, num_, kernel_dim=(3,3)):
    convT_1 = Conv2DTranspose(filter_dim, kernel_dim,strides=(2,2),padding='same',\
                              name = 'transUp_'+str(num_))(input_layer)
    conv_1 = Conv2D(filter_dim, kernel_dim,padding='same',\
                              name = 'transUp_conv_'+str(num_))(convT_1)
    bn_1 = BatchNormalization(name = 'transUp_bn_'+str(num_))(conv_1)
    ac_1 = Activation('relu',name = 'transUp_ac_'+str(num_))(bn_1)
    return ac_1

def TransDown(input_layer, filter_dim, num_1, kernel_dim = (1,1)):
    bn_1 = BatchNormalization(axis = -1)(input_layer)
    relu_1 = Activation('relu',name = 'transDown_ac_'+str(num_1))(bn_1)
    conv_1 = conv_block(relu_1, filter_dim, (1,1), num_1,-1)
    pool_1 = AveragePooling2D(pool_size = (2,2),strides = (2,2), name='pool_' + str(num_1))(conv_1)
    return pool_1
    
def DenseBlock(input_layer, n_layer, filter_dim, num_1):
    box = []
    for ii in range(n_layer):
        layer_y = conv_block(input_layer,filter_dim, (3,3), num_1,ii)
        box.append(layer_y)
        input_layer = concatenate([layer_y,input_layer])
    dense_all = concatenate([tempp for tempp in box],name='dense_' + str(num_1))
    return dense_all

def bottle_neck(input_layer, n_layer, filter_dim, num_1):
    strides_dim = [1,2,3]
    box = []
    for ii in range(len(strides_dim)):
        conv_ = Conv2D(filter_dim,kernel_size = (2,2),dilation_rate = strides_dim[ii],padding='same')(input_layer)
        box.append(conv_)
    neck_all = concatenate([tempp for tempp in box],name='neck_' + str(num_1))
    neck_all = concatenate([neck_all, input_layer])
    return neck_all

def conv_block(input_layer,filter_dim,kernel_size, num_1,num_2):
    bn_1 = BatchNormalization(axis = -1, name='bn_'+str(num_1)+'_'+str(num_2))(input_layer)
    relu_1 = Activation('relu',name='relu_'+str(num_1)+'_'+str(num_2))(bn_1)
    conv_1 = Conv2D(filter_dim, kernel_size = kernel_size, padding = 'same',\
                    kernel_initializer = "he_uniform", name='conv_'+str(num_1)+'_'+str(num_2),kernel_regularizer = kerel_regularizer_)(relu_1)
    drop_1 = Dropout(0.2)(conv_1)      
    return drop_1 

def dense_block_2(input_layer, n_layer, filter_dim, num_1):
    conc_temp = input_layer
    for ii in range(n_layer):
        temp = conv_block(conc_temp, filter_dim, (3,3), num_1, ii)
        conc_temp = concatenate([conc_temp, temp])
    return conc_temp