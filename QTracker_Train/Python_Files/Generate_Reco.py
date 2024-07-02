import os
import sys
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

def generate_e906(n_events, tvt):
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

# Initialize lists to store the data
dimuon_probability=[]
all_predictions = []
tracks = []
truth = []
total_entries = 0
#Generate training data
while(total_entries<10000000):
    try:
        hits, drift, kinematics = generate_e906(200000,"Train")
        tf.keras.backend.clear_session()
        print("Loaded events")
        model = tf.keras.models.load_model('Networks/event_filter')
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(hits,batch_size=225,verbose=0)
        
        hits=hits[predictions[:,3]>0.75]
        drift=drift[predictions[:,3]>0.75]
        kinematics = kinematics[predictions[:,3]>0.75]
        predictions = predictions[predictions[:,3]>0.75]
        dimu_pred = predictions
        print("Filtered Events")
        if(len(hits>0)):
            tf.keras.backend.clear_session()
            
            model = tf.keras.models.load_model('Networks/Track_Finder_All')
            predictions = (np.round(model.predict(hits,verbose=0)*max_ele)).astype(int)
            all_vtx_track = evaluate_finder(hits,drift,predictions)
            print("Found Tracks")

            tf.keras.backend.clear_session()
            
            model=tf.keras.models.load_model('Networks/Reconstruction_All')
            reco_kinematics = model.predict(all_vtx_track,batch_size=8192,verbose=0)

            vertex_reco=np.concatenate((reco_kinematics.reshape((len(reco_kinematics),3,2)),all_vtx_track),axis=1)

            tf.keras.backend.clear_session()
            
            model=tf.keras.models.load_model('Networks/Vertexing_All')
            reco_vertex = model.predict(vertex_reco,batch_size=8192,verbose=0)

            all_vtx_reco_kinematics=np.concatenate((reco_kinematics,reco_vertex),axis=1)

            print("Reconstructed events for all vertices")

            tf.keras.backend.clear_session()
            
            model = tf.keras.models.load_model('Networks/Track_Finder_Z')
            predictions = (np.round(model.predict(hits,verbose=0)*max_ele)).astype(int)
            z_vtx_track = evaluate_finder(hits,drift,predictions)
            print("Found Tracks")


            tf.keras.backend.clear_session()
            
            model=tf.keras.models.load_model('Networks/Reconstruction_Z')
            reco_kinematics = model.predict(z_vtx_track,batch_size=8192,verbose=0)

            vertex_reco=np.concatenate((reco_kinematics.reshape((len(reco_kinematics),3,2)),z_vtx_track),axis=1)

            tf.keras.backend.clear_session()
            
            model=tf.keras.models.load_model('Networks/Vertexing_Z')
            reco_vertex = model.predict(vertex_reco,batch_size=8192,verbose=0)

            z_vtx_reco_kinematics=np.concatenate((reco_kinematics,reco_vertex),axis=1)

            print("Reconstructed events for z vertices")

            tf.keras.backend.clear_session()
            
            model = tf.keras.models.load_model('Networks/Track_Finder_Target')
            predictions = (np.round(model.predict(hits,verbose=0)*max_ele)).astype(int)
            target_vtx_track = evaluate_finder(hits,drift,predictions)            
            print("Found Tracks")

            tf.keras.backend.clear_session()
            
            model=tf.keras.models.load_model('Networks/Reconstruction_Target')
            reco_kinematics = model.predict(target_vtx_track,batch_size=8192,verbose=0)

            target_vtx_reco_kinematics= reco_kinematics

            reco_kinematics = np.concatenate((all_vtx_reco_kinematics,z_vtx_reco_kinematics,target_vtx_reco_kinematics),axis=1)
            
            dimuon_probability.append(dimu_pred)
            all_predictions.append(reco_kinematics)
            truth.append(kinematics)
            tracks.append(np.column_stack((all_vtx_track,z_vtx_track,target_vtx_track)))
            
            np.save(f'Training_Data/{vertex}_tracks_train.npy',np.concatenate(tracks, axis=0))
            np.save(f'Training_Data/{vertex}_kinematics_train.npy',np.concatenate(all_predictions, axis=0))
            np.save(f'Training_Data/{vertex}_truth_train.npy',np.concatenate(truth, axis=0))
            
            print("Reconstructed events for target vertices")
            total_entries += len(hits)
            print(total_entries)
            del hits, drift, reco_kinematics, reco_vertex
        else: print("No events meeting dimuon criteria.")
    except Exception as e:
        pass        
        
# Initialize lists to store the data
dimuon_probability=[]
all_predictions = []
tracks = []
truth = []
total_entries = 0


#Generate validation data
while(total_entries<1000000):
    try:
        hits, drift, kinematics = generate_e906(200000,"Val")
        tf.keras.backend.clear_session()
        print("Loaded events")
        model = tf.keras.models.load_model('Networks/event_filter')
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(hits,batch_size=225,verbose=0)
        
        hits=hits[predictions[:,3]>0.75]
        drift=drift[predictions[:,3]>0.75]
        kinematics = kinematics[predictions[:,3]>0.75]
        predictions = predictions[predictions[:,3]>0.75]
        dimu_pred = predictions
        print("Filtered Events")
        if(len(hits>0)):
            tf.keras.backend.clear_session()
            
            model = tf.keras.models.load_model('Networks/Track_Finder_All')
            predictions = (np.round(model.predict(hits,verbose=0)*max_ele)).astype(int)
            all_vtx_track = evaluate_finder(hits,drift,predictions)
            print("Found Tracks")


            tf.keras.backend.clear_session()
            
            model=tf.keras.models.load_model('Networks/Reconstruction_All')
            reco_kinematics = model.predict(all_vtx_track,batch_size=8192,verbose=0)

            vertex_reco=np.concatenate((reco_kinematics.reshape((len(reco_kinematics),3,2)),all_vtx_track),axis=1)

            tf.keras.backend.clear_session()
            
            model=tf.keras.models.load_model('Networks/Vertexing_All')
            reco_vertex = model.predict(vertex_reco,batch_size=8192,verbose=0)

            all_vtx_reco_kinematics=np.concatenate((reco_kinematics,reco_vertex),axis=1)

            print("Reconstructed events for all vertices")

            tf.keras.backend.clear_session()
            
            model = tf.keras.models.load_model('Networks/Track_Finder_Z')
            predictions = (np.round(model.predict(hits,verbose=0)*max_ele)).astype(int)
            z_vtx_track = evaluate_finder(hits,drift,predictions)
            print("Found Tracks")


            tf.keras.backend.clear_session()
            
            model=tf.keras.models.load_model('Networks/Reconstruction_Z')
            reco_kinematics = model.predict(z_vtx_track,batch_size=8192,verbose=0)

            vertex_reco=np.concatenate((reco_kinematics.reshape((len(reco_kinematics),3,2)),z_vtx_track),axis=1)

            tf.keras.backend.clear_session()
            
            model=tf.keras.models.load_model('Networks/Vertexing_Z')
            reco_vertex = model.predict(vertex_reco,batch_size=8192,verbose=0)

            z_vtx_reco_kinematics=np.concatenate((reco_kinematics,reco_vertex),axis=1)

            print("Reconstructed events for z vertices")

            tf.keras.backend.clear_session()
            
            model = tf.keras.models.load_model('Networks/Track_Finder_Target')
            predictions = (np.round(model.predict(hits,verbose=0)*max_ele)).astype(int)
            target_vtx_track = evaluate_finder(hits,drift,predictions)            
            print("Found Tracks")

            tf.keras.backend.clear_session()
            
            model=tf.keras.models.load_model('Networks/Reconstruction_Target')
            reco_kinematics = model.predict(target_vtx_track,batch_size=8192,verbose=0)

            target_vtx_reco_kinematics= reco_kinematics

            reco_kinematics = np.concatenate((all_vtx_reco_kinematics,z_vtx_reco_kinematics,target_vtx_reco_kinematics),axis=1)
            
            dimuon_probability.append(dimu_pred)
            all_predictions.append(reco_kinematics)
            truth.append(kinematics)
            tracks.append(np.column_stack((all_vtx_track,z_vtx_track,target_vtx_track)))
            
            np.save(f'Training_Data/{vertex}_tracks_val.npy',np.concatenate(tracks, axis=0))
            np.save(f'Training_Data/{vertex}_kinematics_val.npy',np.concatenate(all_predictions, axis=0))
            np.save(f'Training_Data/{vertex}_truth_val.npy',np.concatenate(truth, axis=0))
            
            print("Reconstructed events for target vertices")
            total_entries += len(hits)
            print(total_entries)
            del hits, drift, reco_kinematics, reco_vertex
        else: print("No events meeting dimuon criteria.")
    except Exception as e:
        pass
