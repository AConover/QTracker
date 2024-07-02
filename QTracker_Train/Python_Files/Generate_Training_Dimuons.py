import os
import numpy as np
import uproot
from numba import njit, prange
import random
import tensorflow as tf
import gc
from Python_Files/Common_Functions import *

if len(sys.argv) != 2:
        print("Usage: python script.py <Vertex Distribution>")
        print("Currently supports All, Z, Target, and Dump")
        exit(1)

vertex = sys.argv[1]
root_file_train = f"Root_Files/{vertex}_Train_QA_v2.root"
root_file_val = f"Root_Files/{vertex}_Val_QA_v2.root"
model_name = f"Networks/Track_Finder_{vertex}"

pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics = read_root_file(root_file_train)
pos_events_val, pos_drift_val, pos_kinematics_val, neg_events_val, neg_drift_val, neg_kinematics_val = read_root_file(root_file_val)

pos_events=clean(pos_events).astype(int)
neg_events=clean(neg_events).astype(int)
pos_events_val=clean(pos_events_val).astype(int)
neg_events_val=clean(neg_events_val).astype(int)

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
    

kin_means = np.array([2,0,35,-2,0,35])
kin_stds = np.array([0.6,1.2,10,0.6,1.2,10])

vertex_means=np.array([0,0,-300])
vertex_stds=np.array([10,10,300])

means = np.concatenate((kin_means,vertex_means))
stds = np.concatenate((kin_stds,vertex_stds))


max_ele = [200, 200, 168, 168, 200, 200, 128, 128,  112,  112, 128, 128, 134, 134, 
           112, 112, 134, 134,  20,  20,  16,  16,  16,  16,  16,  16,
        72,  72,  72,  72,  72,  72,  72,  72, 200, 200, 168, 168, 200, 200,
        128, 128,  112,  112, 128, 128, 134, 134, 112, 112, 134, 134,
        20,  20,  16,  16,  16,  16,  16,  16,  72,  72,  72,  72,  72,
        72,  72,  72]

n_train = 0
train_input = []
val_input = []
train_kinematics = []
val_kinematics = []

while(n_train<1e7):
    valin, valdrift, valkinematics = generate_hit_matrices(50000, "Val")
    trainin, traindrift, trainkinematics = generate_hit_matrices(500000, "Train")
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model('Networks/event_filter')
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    val_predictions = probability_model.predict(valin,batch_size = 256, verbose=0)
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model('Networks/event_filter')
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    train_predictions = probability_model.predict(trainin,batch_size = 256, verbose=0)

    trainin=trainin[train_predictions[:,3]>0.75]
    traindrift=traindrift[train_predictions[:,3]>0.75]
    train_kinematics.append(trainkinematics[train_predictions[:,3]>0.75])

    valin=valin[val_predictions[:,3]>0.75]
    valdrift=valdrift[val_predictions[:,3]>0.75]
    val_kinematics.append(valkinematics[val_predictions[:,3]>0.75])

    del val_predictions, train_predictions
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_name)
    predictions = (np.round(model.predict(valin, verbose=0)*max_ele)).astype(int)
    val_input.append(evaluate_finder(valin,valdrift,predictions))
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model('Networks/Track_Finder_All')
    predictions = (np.round(model.predict(trainin, verbose=0)*max_ele)).astype(int)
    train_input.append(evaluate_finder(trainin,traindrift,predictions))
    n_train+=len(trainin)
    del trainin, valin, traindrift, valdrift, predictions
    
    np.save(f'Training_Data/{vertex}_Val_In.npy',np.concatenate((val_input)))
    np.save(f'Training_Data/{vertex}_Val_Out.npy',np.concatenate((val_kinematics)))
    np.save(f'Training_Data/{vertex}_Train_In.npy',np.concatenate((train_input)))
    np.save(f'Training_Data/{vertex}_Train_Out.npy',np.concatenate((train_kinematics)))
    
    print(n_train)


