#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import gc


#Define the means and standard deviations for output normalization
kin_means = np.array([2,0,35,-2,0,35])
kin_stds = np.array([0.6,1.2,10,0.6,1.2,10])
vertex_means=np.array([0,0,-300])
vertex_stds=np.array([10,10,300])

#Define the learning rate and callback
learning_rate_reco=1e-5
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

#Load the pre-generated training data
valin_reco = np.load("Training_Data/Dump_Val_In.npy")
valkinematics = np.load("Training_Data/Dump_Val_Out.npy")
filt = np.max(abs(valin_reco.reshape(len(valin_reco),(136))),axis=1)<1000
valin_reco = valin_reco[filt]
valkinematics = valkinematics[filt]

trainin_reco = np.load("Training_Data/Dump_Train_In.npy")
trainkinematics = np.load("Training_Data/Dump_Train_Out.npy")
filt = np.max(abs(trainin_reco.reshape(len(trainin_reco),(136))),axis=1)<1000
trainin_reco = trainin_reco[filt]
trainkinematics = trainkinematics[filt]

trainkin = trainkinematics[:,:6]
valkin = valkinematics[:,:6]

trainkin = (trainkin-kin_means)/kin_stds
valkin = (valkin-kin_means)/kin_stds

tf.keras.backend.clear_session()
model=tf.keras.models.load_model('Networks/Reconstruction_Dump')
optimizer = tf.keras.optimizers.Adam(learning_rate_reco)
model.compile(optimizer=optimizer,
      loss=tf.keras.losses.mse,
      metrics=tf.keras.metrics.RootMeanSquaredError())
val_loss_before=model.evaluate(valin_reco,valkin,batch_size=100,verbose=2)[0]
print(val_loss_before)
history = model.fit(trainin_reco, trainkin,
            epochs=10000, batch_size=1024, verbose=2, validation_data=(valin_reco,valkin),callbacks=[callback])
model.save('Networks/Reconstruction_Dump')
