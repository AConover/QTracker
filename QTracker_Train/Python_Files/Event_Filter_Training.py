import os
import numpy as np
import uproot
from numba import njit, prange
import random
import tensorflow as tf
import gc

def read_root_file(root_file):
    targettree = uproot.open(root_file + ':QA_ana')
    targetevents = len(targettree['n_tracks'].array(library='np'))

    detector_data = read_detector_data(targettree)

    pos_events = np.zeros((targetevents, 54))
    pos_drift = np.zeros((targetevents, 30))
    pos_kinematics = np.zeros((targetevents, 6))
    neg_events = np.zeros((targetevents, 54))
    neg_drift = np.zeros((targetevents, 30))
    neg_kinematics = np.zeros((targetevents, 6))
    
    process_target_events(targetevents, detector_data, pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics)
    
    return pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics

def read_detector_data(targettree):
    detector_data = {}
    for key in ['D0U_ele', 'D0Up_ele', 'D0X_ele', 'D0Xp_ele', 'D0V_ele', 'D0Vp_ele', 
                'D2U_ele', 'D2Up_ele', 'D2X_ele', 'D2Xp_ele', 'D2V_ele', 'D2Vp_ele', 
                'D3pU_ele', 'D3pUp_ele', 'D3pX_ele', 'D3pXp_ele', 'D3pV_ele', 'D3pVp_ele', 
                'D3mU_ele', 'D3mUp_ele', 'D3mX_ele', 'D3mXp_ele', 'D3mV_ele', 'D3mVp_ele', 
                'H1B_ele', 'H1T_ele', 'H1L_ele', 'H1R_ele', 'H2L_ele', 'H2R_ele', 'H2B_ele', 'H2T_ele', 
                'H3B_ele', 'H3T_ele', 'H4Y1L_ele', 'H4Y1R_ele', 'H4Y2L_ele', 'H4Y2R_ele', 'H4B_ele', 'H4T_ele', 
                'P1Y1_ele', 'P1Y2_ele', 'P1X1_ele', 'P1X2_ele', 'P2X1_ele', 'P2X2_ele', 'P2Y1_ele', 'P2Y2_ele', 
                'gpx', 'gpy', 'gpz', 'gvx', 'gvy', 'gvz', 'pid']:
        detector_data[key] = targettree[key].array(library='np')
    return detector_data

def process_target_events(targetevents, detector_data, pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics):
    pid = detector_data['pid']
    gpx = detector_data['gpx']
    gpy = detector_data['gpy']
    gpz = detector_data['gpz']
    gvx = detector_data['gvx']
    gvy = detector_data['gvy']
    gvz = detector_data['gvz']
    
    for j in range(targetevents):
        first = pid[j][0]
        pos, neg = (0, 1) if first > 0 else (1, 0)
        
        fill_kinematics(gpx, gpy, gpz, gvx, gvy, gvz, pos_kinematics, neg_kinematics, j, pos, neg)
        fill_events_and_drift(detector_data, pos_events, pos_drift, neg_events, neg_drift, j, pos, neg)

def fill_kinematics(gpx, gpy, gpz, gvx, gvy, gvz, pos_kinematics, neg_kinematics, j, pos, neg):
    pos_kinematics[j][0] = gpx[j][pos]
    pos_kinematics[j][1] = gpy[j][pos]
    pos_kinematics[j][2] = gpz[j][pos]
    pos_kinematics[j][3] = gvx[j][pos]
    pos_kinematics[j][4] = gvy[j][pos]
    pos_kinematics[j][5] = gvz[j][pos]
    neg_kinematics[j][0] = gpx[j][neg]
    neg_kinematics[j][1] = gpy[j][neg]
    neg_kinematics[j][2] = gpz[j][neg]
    neg_kinematics[j][3] = gvx[j][neg]
    neg_kinematics[j][4] = gvy[j][neg]
    neg_kinematics[j][5] = gvz[j][neg]

def fill_events_and_drift(detector_data, pos_events, pos_drift, neg_events, neg_drift, j, pos, neg):
    for k in range(54):
        pos_events[j][k] = detector_data[f'key_{k}_ele'][j][pos]
        neg_events[j][k] = detector_data[f'key_{k}_ele'][j][neg]
    for k in range(30):
        pos_drift[j][k] = detector_data[f'key_{k}_drift'][j][pos]
        neg_drift[j][k] = detector_data[f'key_{k}_drift'][j][neg]


# Reading training and validation data from ROOT files
pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics = read_root_file('Root_Files/Target_Train_QA_v2.root')
pos_events_val, pos_drift_val, pos_kinematics_val, neg_events_val, neg_drift_val, neg_kinematics_val = read_root_file('Root_Files/Target_Val_QA_v2.root')


# Clean event data by setting values > 1000 to 0.
@njit(parallel=True)
def clean(events):
    for j in prange(len(events)):
        for i in prange(54):
            if(events[j][i]>1000):
                events[j][i]=0
    return events

pos_events=clean(pos_events).astype(int)
neg_events=clean(neg_events).astype(int)
pos_events_val=clean(pos_events_val).astype(int)
neg_events_val=clean(neg_events_val).astype(int)

@njit(parallel=True)
def track_injection(hits,pos_e,neg_e):
    # Inject tracks into the hit matrices
    category=np.zeros((len(hits)))
    track=np.zeros((len(hits),108))
    for z in prange(len(hits)):
        m = random.randrange(0,2)
        j=random.randrange(len(pos_e))
        for k in range(54):
            if(pos_e[j][k]>0):
                if(random.random()<m*0.94) or ((k>29)&(k<45)):
                    hits[z][k][int(pos_e[j][k]-1)]=1
                track[z][k]=pos_e[j][k]
            if(neg_e[j][k]>0):
                if(random.random()<m*0.94) or ((k>29)&(k<45)):
                    hits[z][k][int(neg_e[j][k]-1)]=1
        category[z]=m        

    return hits,category


@njit()
def hit_matrix(detectorid,elementid,hits,station): #Convert into hit matrices
    for j in range (len(detectorid)):
        rand = random.random()
        #St 1
        if(station==1) and (rand<0.85):
            if ((detectorid[j]<7) or (detectorid[j]>30)) and (detectorid[j]<35):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
        #St 2
        elif(station==2):
            if (detectorid[j]>12 and (detectorid[j]<19)) or ((detectorid[j]>34) and (detectorid[j]<39)):
                if((detectorid[j]<15) and (rand<0.76)) or ((detectorid[j]>14) and (rand<0.86)) or (detectorid[j]==17):
                    hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
        #St 3
        elif(station==3) and (rand<0.8):
            if (detectorid[j]>18 and (detectorid[j]<31)) or ((detectorid[j]>38) and (detectorid[j]<47)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
        #St 4
        elif(station==4):
            if ((detectorid[j]>40) and (detectorid[j]<55)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
        if(rand<0.25):
            if ((detectorid[j]>40) and (detectorid[j]<47)): 
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
    return hits


def generate_e906(n_events, tvt):
    #Import NIM3 events and put them on the hit matrices.
    #Randomly choosing between 1 and 6 events per station gets approximately the correct occupancies.
    filelist=['output_part1.root:tree_nim3','output_part2.root:tree_nim3','output_part3.root:tree_nim3',
             'output_part4.root:tree_nim3','output_part5.root:tree_nim3','output_part6.root:tree_nim3',
             'output_part7.root:tree_nim3','output_part8.root:tree_nim3','output_part9.root:tree_nim3']
    targettree = uproot.open("NIM3/"+random.choice(filelist))
    detectorid_nim3=targettree["det_id"].arrays(library="np")["det_id"]
    elementid_nim3=targettree["ele_id"].arrays(library="np")["ele_id"]
    hits = np.zeros((n_events,54,201))
    for n in range (n_events): #Create NIM3 events
        g=random.choice([0,1,2,3,4,5,6])
        for m in range(g):
            i=random.randrange(len(detectorid_nim3))
            hits[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],hits[n],1)
            i=random.randrange(len(detectorid_nim3))
            hits[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],hits[n],2)
            i=random.randrange(len(detectorid_nim3))
            hits[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],hits[n],3)
            i=random.randrange(len(detectorid_nim3))
            hits[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],hits[n],4)
    del detectorid_nim3, elementid_nim3
    
    #Place the full tracks that are reconstructable
    if(tvt=="Train"):
        hits,category=track_injection(hits,pos_events,neg_events)    
    if(tvt=="Val"):
        hits,category=track_injection(hits,pos_events_val,neg_events_val)    
    return hits.astype(bool), category.astype(int)

# Set learning rate and callback for early stopping
learning_rate_filter = 1e-6
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
n_train = 0

print("Before while loop:", n_train)
while n_train < 1e7:
    # Generate training and validation data
    trainin, trainsignals = generate_e906(500000, "Train")
    n_train += len(trainin)
    print("Generated Training Data")
    valin, valsignals = generate_e906(50000, "Val")
    print("Generated Validation Data")
    
    # Clear session and reset TensorFlow graph
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()

    # Load and compile the model
    model = tf.keras.models.load_model('Networks/event_filter')
    optimizer = tf.keras.optimizers.Adam(learning_rate_filter)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Evaluate the model before training
    val_loss_before = model.evaluate(valin, valsignals, batch_size=256, verbose=2)[0]

    # Train the model
    history = model.fit(trainin, trainsignals,
                        epochs=1000, batch_size=256, verbose=2, validation_data=(valin, valsignals), callbacks=[callback])

    # Check if the validation loss improved
    if min(history.history['val_loss']) < val_loss_before:
        model.save('Networks/event_filter')
        learning_rate_filter *= 2
    learning_rate_filter /= 2

    print('\n')

    # Clear session and force garbage collection to release GPU memory
    tf.keras.backend.clear_session()
    del trainsignals, trainin, valin, valsignals, model
    gc.collect()
    print(n_train)

