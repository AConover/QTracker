import os
import numpy as np
import uproot
from numba import njit, prange
import random
import tensorflow as tf
import gc
import sys
from Python_Files/Common_Functions import *

if len(sys.argv) != 2:
        print("Usage: python script.py <Vertex Distribution>")
        print("Currently supports All_Vertex, Beamline, Target, and Dump")
        exit(1)

if sys.argv[1] == "All_Vertex":
    root_file_train = "Root_Files/All_Vertex_Train_QA_v2.root"
    root_file_val = "Root_Files/All_Vertex_Val_QA_v2.root"
    model_name = "Networks/Track_Finder_All"
elif sys.argv[1] == "Beamline":
    root_file_train = "Root_Files/Z_Train_QA_v2.root"
    root_file_val = "Root_Files/Z_Val_QA_v2.root"
    model_name = "Networks/Track_Finder_Z"
elif sys.argv[1] == "Target":
    root_file_train = "Root_Files/Target_Train_QA_v2.root"
    root_file_val = "Root_Files/Target_Val_QA_v2.root"
    model_name = "Networks/Track_Finder_Target"
elif sys.argv[1] == "Dump":
    root_file_train = "Root_Files/Dump_Train_QA_v2.root"
    root_file_val = "Root_Files/Dump_Val_QA_v2.root"
    model_name = "Networks/Track_Finder_Dump"
else:
    print("Unrecognized option. Quitting...")
    exit(1)

pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics = read_root_file(root_file_train)
pos_events_val, pos_drift_val, pos_kinematics_val, neg_events_val, neg_drift_val, neg_kinematics_val = read_root_file(root_file_val)

del pos_drift, neg_drift, pos_kinematics, neg_kinematics
del pos_drift_val, neg_drift_val, pos_kinematics_val, neg_kinematics_val

pos_events=clean(pos_events).astype(int)
neg_events=clean(neg_events).astype(int)
pos_events_val=clean(pos_events_val).astype(int)
neg_events_val=clean(neg_events_val).astype(int)

@njit(parallel=True)
def track_injection(hits, pos_e, neg_e):
    n_events = len(hits)
    track_real = np.zeros((n_events, 68), dtype=np.float32) 

    for z in prange(n_events):
        j = np.random.randint(len(pos_e))
        for k in range(54):
            pos_val = pos_e[j][k]
            neg_val = neg_e[j][k]
            if pos_val > 0 and (np.random.random() < 0.94 or k > 29):
                hits[z][k][int(pos_val - 1)] = 1
            if neg_val > 0 and (np.random.random() < 0.94 or k > 29):
                hits[z][k][int(neg_val - 1)] = 1

        # Convert the hits into tracks to be reconstructed.
        track_real[z, :6] = pos_e[j, :6]  
        track_real[z, 6:12] = pos_e[j, 12:18]
        track_real[z, 34:40] = neg_e[j, :6]  
        track_real[z, 40:46] = neg_e[j, 12:18]
        # St. 3p gets positive values, St. 3m gets negative values.
        track_real[z, 12:18] = np.where((pos_e[j, 18]) > 0, pos_e[j, 18:24], -pos_e[j, 24:30])
        track_real[z, 46:52] = np.where(neg_e[j, 18] > 0, neg_e[j, 18:24], -neg_e[j, 24:30])
        # Pairs of hodoscopes are mutually exclusive, this gives positive or negative values depending on the array.
        track_real[z, 18:26] = np.where(pos_e[j, 30:45:2] > 0, pos_e[j, 30:45:2], -pos_e[j, 31:46:2])
        track_real[z, 52:60] = np.where(neg_e[j, 30:45:2] > 0 , neg_e[j, 30:45:2], -neg_e[j, 31:46:2])
        track_real[z, 26:34] = pos_e[j, 46:54]
        track_real[z, 60:68] = neg_e[j, 46:54]

    return hits, track_real

def generate_hit_matrices(n_events, tvt):
    #Create the realistic background for events
    hits, _ = build_background(n_events)
    #Place the full tracks that are reconstructable
    if(tvt=="Train"):
        hits,track=track_injection(hits,pos_events,neg_events)    
    if(tvt=="Val"):
        hits,track=track_injection(hits,pos_events_val,neg_events_val)    
    return hits.astype(bool), track.astype(int)
    
max_ele = [200, 200, 168, 168, 200, 200, 128, 128,  112,  112, 128, 128, 134, 134, 
           112, 112, 134, 134,  20,  20,  16,  16,  16,  16,  16,  16,
        72,  72,  72,  72,  72,  72,  72,  72, 200, 200, 168, 168, 200, 200,
        128, 128,  112,  112, 128, 128, 134, 134, 112, 112, 134, 134,
        20,  20,  16,  16,  16,  16,  16,  16,  72,  72,  72,  72,  72,
        72,  72,  72]

learning_rate_finder=1e-6
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
n_train=0

print("Before while loop:", n_train)
while(n_train<1e7):
    trainin, traintrack = generate_hit_matrices(750000, "Train")
    print("Generated Training Data")
    traintrack = traintrack/max_ele
    
    valin, valtrack = generate_hit_matrices(75000, "Val")
    print("Generated Validation Data")
    valtrack = valtrack/max_ele
    
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.load_model('Networks/event_filter')
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    val_predictions = probability_model.predict(valin,batch_size=225)
    train_predictions = probability_model.predict(trainin,batch_size=225)


    trainin=trainin[train_predictions[:,3]>0.75]
    traintrack=traintrack[train_predictions[:,3]>0.75]

    valin=valin[val_predictions[:,3]>0.75]
    valtrack=valtrack[val_predictions[:,3]>0.75]
    
    n_train+=len(trainin)
    
    del train_predictions, val_predictions
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    # Specify the optimizer, and compile the model with loss functions for both outputs
    model = tf.keras.models.load_model(model_name)
    optimizer = tf.keras.optimizers.Adam(learning_rate_finder)
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.mse,
              metrics=tf.keras.metrics.RootMeanSquaredError())

    val_loss_before=model.evaluate(valin,valtrack,batch_size=100,verbose=2)[0]
    print(val_loss_before)
    history = model.fit(trainin, traintrack,
                    epochs=1000, batch_size=100, verbose=2, validation_data=(valin,valtrack),callbacks=[callback])
    if(min(history.history['val_loss'])<val_loss_before):
        model.save(model_name)
        learning_rate_finder=learning_rate_finder*2
    learning_rate_finder=learning_rate_finder/2
    tf.keras.backend.clear_session()
    del trainin, valin, traintrack, valtrack,model
    gc.collect()  # Force garbage collection to release GPU memory
    print(n_train)


