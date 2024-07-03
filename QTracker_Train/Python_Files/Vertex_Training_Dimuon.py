import numpy as np
import tensorflow as tf
import gc

if len(sys.argv) != 2:
        print("Usage: python script.py <Vertex Distribution>")
        print("Currently supports All, Z, Target, and Dump")
        exit(1)

vertex = sys.argv[1]

#Define the means and standard deviations for output normalization
vertex_means=np.array([0,0,-300])
vertex_stds=np.array([10,10,300])

#Define the learning rate and callback
learning_rate_vertex=1e-6
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

#Load the pre-generated training data
valin_reco = np.load(f"Training_Data/{vertex}_Val_In.npy")
valkinematics = np.load(f"Training_Data/{vertex}_Val_Out.npy")
filt = np.max(abs(valin_reco.reshape(len(valin_reco),(136))),axis=1)<1000
valin_reco = valin_reco[filt]
valkinematics = valkinematics[filt]

trainin_reco = np.load("f"Training_Data/{vertex}_Train_In.npy")
trainkinematics = np.load("f"Training_Data/{vertex}_Train_Out.npy")
filt = np.max(abs(trainin_reco.reshape(len(trainin_reco),(136))),axis=1)<1000
trainin_reco = trainin_reco[filt]
trainkinematics = trainkinematics[filt]

trainvertex = trainkinematics[:,6:]
valvertex = valkinematics[:,6:]

trainvertex = (trainvertex-vertex_means)/vertex_stds
valvertex = (valvertex-vertex_means)/vertex_stds

model=tf.keras.models.load_model(f'Networks/Reconstruction_{vertex}')

train_reco = model.predict(trainin_reco)
val_reco = model.predict(valin_reco)

train_input=np.concatenate((train_reco.reshape((len(train_reco),3,2)),trainin_reco),axis=1)
val_input=np.concatenate((val_reco.reshape((len(val_reco),3,2)),valin_reco),axis=1)

tf.keras.backend.clear_session()
model=tf.keras.models.load_model('Networks/Vertexing_{vertex}')
optimizer = tf.keras.optimizers.Adam(learning_rate_vertex)
model.compile(optimizer=optimizer,
      loss=tf.keras.losses.mse,
      metrics=tf.keras.metrics.RootMeanSquaredError())
val_loss_before=model.evaluate(val_input,valvertex,batch_size=100,verbose=2)[0]
print(val_loss_before)
history = model.fit(train_input, trainvertex,
            epochs=10000, batch_size=1024, verbose=2, validation_data=(val_input,valvertex),callbacks=[callback])
model.save(f'Networks/Reconstruction_{vertex}')
