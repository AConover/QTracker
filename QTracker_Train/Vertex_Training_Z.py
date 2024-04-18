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
learning_rate_vertex=1e-5
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

#Load the pre-generated training data
valin_reco = np.load("Z_Val_In.npy")
trainin_reco = np.load("Z_Train_In.npy")

valkinematics = np.load("Z_Val_Out.npy")
trainkinematics = np.load("Z_Train_Out.npy")

trainvertex = trainkinematics[:,6:]
valvertex = valkinematics[:,6:]

trainvertex = (trainvertex-vertex_means)/vertex_stds
valvertex = (valvertex-vertex_means)/vertex_stds

model=tf.keras.models.load_model('Reconstruction_Z')

train_reco = model.predict(trainin_reco,batch_size=8192)
val_reco = model.predict(valin_reco,batch_size=8192)

train_input=np.concatenate((train_reco.reshape((len(train_reco),3,2)),trainin_reco),axis=1)
val_input=np.concatenate((val_reco.reshape((len(val_reco),3,2)),valin_reco),axis=1)

tf.keras.backend.clear_session()
model=tf.keras.models.load_model('Vertexing_Z')
optimizer = tf.keras.optimizers.Adam(learning_rate_vertex)
model.compile(optimizer=optimizer,
      loss=tf.keras.losses.mse,
      metrics=tf.keras.metrics.RootMeanSquaredError())
val_loss_before=model.evaluate(val_input,valvertex,batch_size=100,verbose=2)[0]
print(val_loss_before)
history = model.fit(train_input, trainvertex,
            epochs=10000, batch_size=1024, verbose=2, validation_data=(val_input,valvertex),callbacks=[callback])
model.save('Vertexing_Z')
