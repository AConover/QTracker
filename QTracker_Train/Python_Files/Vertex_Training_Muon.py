import numpy as np
import tensorflow as tf
import gc

if len(sys.argv) != 2:
        print("Usage: python script.py <charge>")
        print("Options are Pos and Neg")
        exit(1)

charge = sys.argv[1]

#Define the learning rate and callback
learning_rate_vertex=1e-6
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

#Load the pre-generated training data
valin_reco = np.load("Training_Data/Muon_Val_In.npy")
valkinematics = np.load("Training_Data/Muon_Val_Out.npy")
filt = np.max(abs(valin_reco.reshape(len(valin_reco),(136))),axis=1)<1000
valin_reco = valin_reco[filt]
valkinematics = valkinematics[filt]

trainin_reco = np.load("Training_Data/Muon_Train_In.npy")
trainkinematics = np.load("Training_Data/Muon_Train_Out.npy")
filt = np.max(abs(trainin_reco.reshape(len(trainin_reco),(136))),axis=1)<1000
trainin_reco = trainin_reco[filt]
trainkinematics = trainkinematics[filt]

if(charge=="Pos"):
    trainvertex = trainkinematics[:,0]
    valvertex = valkinematics[:,0]
if(charge=="Neg"):
    trainvertex = trainkinematics[:,1]
    valvertex = valkinematics[:,1]

tf.keras.backend.clear_session()
model=tf.keras.models.load_model(f'Networks/Vertexing_{charge}')
optimizer = tf.keras.optimizers.Adam(learning_rate_vertex)
model.compile(optimizer=optimizer,
      loss=tf.keras.losses.mse,
      metrics=tf.keras.metrics.RootMeanSquaredError())
val_loss_before=model.evaluate(val_input,valvertex,batch_size=100,verbose=2)[0]
print(val_loss_before)
history = model.fit(train_input, trainvertex,
            epochs=10000, batch_size=1024, verbose=2, validation_data=(val_input,valvertex),callbacks=[callback])
model.save(f'Networks/Reconstruction_{vertex}')
