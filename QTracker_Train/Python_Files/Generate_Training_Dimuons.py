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
        print("Currently supports All, Z, Target, and Dump")
        exit(1)

vertex = sys.argv[1]
root_file_train = f"Root_Files/{vertex}_Train_QA_v2.root"
root_file_val = f"Root_Files/{vertex}_Val_QA_v2.root"
model_name = f"Networks/Track_Finder_{vertex}"

@njit(parallel=True)
def track_injection(hits,drift,pos_e,neg_e,pos_d,neg_d,pos_k,neg_k):
    #Start generating the events
    kin=np.zeros((len(hits),9))
    for z in prange(len(hits)):
        j=random.randrange(len(pos_e))
        kin[z, :3] = pos_k[j, :3]
        kin[z, 3:9] = neg_k[j]
        for k in range(54):
            if(pos_e[j][k]>0):
                if(random.random()<0.94) and (k<30):
                    hits[z][k][int(pos_e[j][k]-1)]=1
                    drift[z][k][int(pos_e[j][k]-1)]=pos_d[j][k]
                if(k>29):
                    hits[z][k][int(pos_e[j][k]-1)]=1
            if(neg_e[j][k]>0):
                if(random.random()<0.94) and (k<30):
                    hits[z][k][int(neg_e[j][k]-1)]=1
                    drift[z][k][int(neg_e[j][k]-1)]=neg_d[j][k]
                if(k>29):
                    hits[z][k][int(neg_e[j][k]-1)]=1

    return hits,drift,kin

def generate_hit_matrices(n_events, tvt):
    #Create the realistic background for events
    hits, drift = build_background(n_events)
    #Place the full tracks that are reconstructable
    if(tvt=="Train"):
        hits,drift,kinematics=track_injection(hits,drift,pos_events,neg_events,pos_drift,neg_drift,pos_kinematics,neg_kinematics)    
    if(tvt=="Val"):
        hits,drift,kinematics=track_injection(hits,drift,pos_events_val,neg_events_val,pos_drift_val,neg_drift_val,pos_kinematics_val,neg_kinematics_val)    
    return hits.astype(bool), drift, kinematics

pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics = read_root_file(root_file_train)
pos_events_val, pos_drift_val, pos_kinematics_val, neg_events_val, neg_drift_val, neg_kinematics_val = read_root_file(root_file_val)

pos_events=clean(pos_events).astype(int)
neg_events=clean(neg_events).astype(int)
pos_events_val=clean(pos_events_val).astype(int)
neg_events_val=clean(neg_events_val).astype(int)

n_train = 0

# Initialize lists to store data
train_input = []
train_kinematics = []
val_input = []
val_kinematics = []

# Detect the number of GPUs available
gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
print(f"Number of GPUs available: {num_gpus}")

# Set up strategy for distributed training
if num_gpus > 1:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()

# Adjust batch size for the number of GPUs
batch_size_ef = 256 * num_gpus
batch_size_tf = 64 * num_gpus

while n_train < 1e7:
    valin, valdrift, valkinematics = generate_hit_matrices(50000, "Val")
    trainin, traindrift, trainkinematics = generate_hit_matrices(500000, "Train")
        
    # Clear session and load the probability model for event filtering
    tf.keras.backend.clear_session()
    with strategy.scope():
        probability_model = tf.keras.Sequential([tf.keras.models.load_model('Networks/event_filter'), tf.keras.layers.Softmax()])
        train_predictions = probability_model.predict(trainin, batch_size=batch_size_ef,verbose=0)
        val_predictions = probability_model.predict(valin, batch_size=batch_size_ef,verbose=0)
        train_mask = train_predictions[:, 1] > 0.75
        val_mask = val_predictions[:, 1] > 0.75

    # Clear session and load track finder models
    tf.keras.backend.clear_session()
    with strategy.scope():
        track_finder_pos = tf.keras.models.load_model('Networks/Track_Finder_Pos')
        pos_predictions_val = (np.round(track_finder_pos.predict(valin, verbose=0, batch_size = batch_size_tf) * max_ele)).astype(int)
        pos_predictions_train = (np.round(track_finder_pos.predict(trainin, verbose=0, batch_size = batch_size_tf) * max_ele)).astype(int)
    with strategy.scope():
        tf.keras.backend.clear_session()
        track_finder_neg = tf.keras.models.load_model('Networks/Track_Finder_Neg')
        neg_predictions_val = (np.round(track_finder_neg.predict(valin, verbose=0, batch_size = batch_size_tf) * max_ele)).astype(int)
        neg_predictions_train = (np.round(track_finder_neg.predict(trainin, verbose=0, batch_size = batch_size_tf) * max_ele)).astype(int)
    
    #Update mask for validation data
    track_val = evaluate_finder(valin, valdrift, np.column_stack((pos_predictions_val, neg_predictions_val)))
    results_val = calc_mismatches(track_val)
    val_mask &= ((results_val[0::4] < 2) & (results_val[1::4] < 2) & (results_val[2::4] < 3) & (results_val[3::4] < 3)).all(axis=0)
    
    #Update mask for training data
    track_train = evaluate_finder(trainin, traindrift, np.column_stack((pos_predictions_train, neg_predictions_train)))
    results_train = calc_mismatches(track_train)
    train_mask &= ((results_train[0::4] < 2) & (results_train[1::4] < 2) & (results_train[2::4] < 3) & (results_train[3::4] < 3)).all(axis=0)

    # Apply masks
    trainin = trainin[train_mask]
    traindrift = traindrift[train_mask]
    trainkinematics = trainkinematics[train_mask]
    valin = valin[val_mask]
    valdrift = valdrift[val_mask]
    valkinematics = valkinematics[val_mask]
    
    # Clear session and load the track finder model
    tf.keras.backend.clear_session()
    with strategy.scope():
        track_finder_model = tf.keras.models.load_model(model_name)
        val_predictions = (np.round(track_finder_model.predict(valin, verbose=0, batch_size = batch_size_tf) * max_ele)).astype(int)
        track_val = evaluate_finder(valin, valdrift, val_predictions)
        results_val = calc_mismatches(track_val)
        val_mask = ((results_val[0::4] < 2) & (results_val[1::4] < 2) & (results_val[2::4] < 3) & (results_val[3::4] < 3)).all(axis=0)

        train_predictions = (np.round(track_finder_model.predict(trainin, verbose=0, batch_size = batch_size_tf) * max_ele)).astype(int)
        track_train = evaluate_finder(trainin, traindrift, train_predictions)
        results_train = calc_mismatches(track_train)
        train_mask = ((results_train[0::4] < 2) & (results_train[1::4] < 2) & (results_train[2::4] < 3) & (results_train[3::4] < 3)).all(axis=0)
    
    val_input.append(track_val[val_mask])
    train_input.append(track_train[train_mask])
    
    # Append data to lists
    train_kinematics.append(trainkinematics[train_mask])
    val_kinematics.append(valkinematics[val_mask])

    n_train = len(np.concatenate(train_kinematics))

    # Save (use a consistent saving strategy to avoid repeated concatenation)
    np.save(f'Training_Data/{vertex}_Val_In.npy', np.concatenate(val_input))  
    np.save(f'Training_Data/{vertex}_Val_Out.npy', np.concatenate(val_kinematics))
    np.save(f'Training_Data/{vertex}_Train_In.npy', np.concatenate(train_input))
    np.save(f'Training_Data/{vertex}_Train_Out.npy', np.concatenate(train_kinematics))
    gc.collect()
    print(n_train)



