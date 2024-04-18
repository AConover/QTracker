import numpy as np
import uproot
import numba
from numba import njit, prange
import random
import tensorflow as tf
import gc

@numba.jit(nopython=True)
def hit_matrix(detectorid,elementid,drifttime,hits,drift,station): #Convert into hit matrices
    for j in range (len(detectorid)):
        rand = random.random()
        #St 1
        if(station==1) and (rand<0.85):
            if ((detectorid[j]<7) or (detectorid[j]>30)) and (detectorid[j]<35):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
        #St 2
        elif(station==2):
            if (detectorid[j]>12 and (detectorid[j]<19)) or ((detectorid[j]>34) and (detectorid[j]<39)):
                if((detectorid[j]<15) and (rand<0.76)) or ((detectorid[j]>14) and (rand<0.86)) or (detectorid[j]==17):
                    hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                    drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
        #St 3
        elif(station==3) and (rand<0.8):
            if (detectorid[j]>18 and (detectorid[j]<31)) or ((detectorid[j]==39) or (detectorid[j]==40)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
        #St 4
        elif(station==4):
            if ((detectorid[j]>39) and (detectorid[j]<55)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
    return hits,drift

@numba.jit(nopython=True)
def hit_matrix_partial_track(detectorid,elementid,drifttime,hits,drift,station): #Convert into hit matrices
    for j in prange (len(detectorid)):
        #St 1
        if(station==1):
            if ((detectorid[j]<7) or (detectorid[j]>30)) and (detectorid[j]<35):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
        #St 2
        elif(station==2):
            if (detectorid[j]>12 and (detectorid[j]<19)) or ((detectorid[j]>34) and (detectorid[j]<39)):
                if((detectorid[j]<15) and (rand<0.76)) or ((detectorid[j]>14) and (rand<0.86)) or (detectorid[j]==17):
                    hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                    drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
        #St 3
        elif(station==3):
            if (detectorid[j]>18 and (detectorid[j]<31)) or ((detectorid[j]>38) and (detectorid[j]<47)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
        #St 4
        elif(station==4):
            if ((detectorid[j]>40) and (detectorid[j]<55)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
    return hits,drift

@numba.jit(nopython=True)
def hit_matrix_mc(detectorid,elementid,drifttime,hits,drift): #Convert into hit matrices
    for j in prange (len(detectorid)):
        if ((detectorid[j]<7) or ((detectorid[j]>12) and (detectorid[j]<55))) and ((random.random()<0.94) or (detectorid[j]>30)):
            hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
            drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
    return hits,drift

# Function to evaluate the Track Finder neural network.
@njit(parallel=True)
def evaluate_finder(testin, testdrift, predictions):
    # The function constructs inputs for the neural network model based on test data
    # and predictions, processing each event in parallel for efficiency.
    reco_in = np.zeros((len(testin), 68, 2))
    
    def process_entry(i, dummy, j_offset):
        j = dummy if dummy <= 5 else dummy + 6
        if dummy > 11:
            if predictions[i][12+j_offset] > 0:
                j = dummy + 6
            elif predictions[i][12+j_offset] < 0:
                j = dummy + 12

        if dummy > 17:
            j = 2 * (dummy - 18) + 30 if predictions[i][2 * (dummy - 18) + 30 + j_offset] > 0 else 2 * (dummy - 18) + 31

        if dummy > 25:
            j = dummy + 20

        k = abs(predictions[i][dummy + j_offset])
        sign = k / predictions[i][dummy + j_offset] if k > 0 else 1
        if(dummy<6):window=15
        elif(dummy<12):window=5
        elif(dummy<18):window=5
        elif(dummy<26):window=1
        else:window=3
        k_sum = np.sum(testin[i][j][k - window:k + window-1])
        if k_sum > 0 and ((dummy < 18) or (dummy > 25)):
            k_temp = k
            n = 1
            while testin[i][j][k - 1] == 0:
                k_temp += n
                n = -n * (abs(n) + 1) / abs(n)
                if 0 <= k_temp < 201:
                    k = int(k_temp)

        reco_in[i][dummy + j_offset][0] = sign * k
        reco_in[i][dummy + j_offset][1] = testdrift[i][j][k - 1]

    for i in prange(predictions.shape[0]):
        for dummy in prange(34):
            process_entry(i, dummy, 0)
        
        for dummy in prange(34):
            process_entry(i, dummy, 34)      

    return reco_in

max_ele = [200, 200, 168, 168, 200, 200, 128, 128,  112,  112, 128, 128, 134, 134, 
           112, 112, 134, 134,  20,  20,  16,  16,  16,  16,  16,  16,
        72,  72,  72,  72,  72,  72,  72,  72, 200, 200, 168, 168, 200, 200,
        128, 128,  112,  112, 128, 128, 134, 134, 112, 112, 134, 134,
        20,  20,  16,  16,  16,  16,  16,  16,  72,  72,  72,  72,  72,
        72,  72,  72]

def generate_e906(n_events,tvt):
    #Import NIM3 events and put them on the hit matrices.
    #Randomly choose between 1 and 6 events per station.
    filelist=['output_part1.root:tree_nim3','output_part2.root:tree_nim3','output_part3.root:tree_nim3',
             'output_part4.root:tree_nim3','output_part5.root:tree_nim3','output_part6.root:tree_nim3',
             'output_part7.root:tree_nim3']
    targettree = uproot.open("NIM3/"+random.choice(filelist))
    detectorid_nim3=targettree["det_id"].arrays(library="np")["det_id"]
    elementid_nim3=targettree["ele_id"].arrays(library="np")["ele_id"]
    driftdistance_nim3=targettree["drift_dist"].arrays(library="np")["drift_dist"]
    hits = np.zeros((n_events,54,201))
    drift = np.zeros((n_events,54,201))
    kinematics = np.zeros((n_events,9))
    for n in range (n_events): #Create NIM3 events
        g=random.choice([1,2,3,4,5,6])
        for m in range(g):
            i=random.randrange(len(detectorid_nim3))
            hits[n],drift[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],driftdistance_nim3[i],hits[n],drift[n],1)
            i=random.randrange(len(detectorid_nim3))
            hits[n],drift[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],driftdistance_nim3[i],hits[n],drift[n],2)
            i=random.randrange(len(detectorid_nim3))
            hits[n],drift[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],driftdistance_nim3[i],hits[n],drift[n],3)
            i=random.randrange(len(detectorid_nim3))
            hits[n],drift[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],driftdistance_nim3[i],hits[n],drift[n],4)
    del detectorid_nim3, elementid_nim3,driftdistance_nim3
      
    #Place Full Tracks on the hit matrix
    if(tvt=="Train"):
        targettree = uproot.open("Root_Files/Target_Train_QA.root:QA_ana")
    if(tvt=="Val"):
        targettree = uproot.open("Root_Files/Target_Val_QA.root:QA_ana")
    if(tvt=="Test"):
        targettree = uproot.open("Root_Files/Target_Test_QA.root:QA_ana")
    detectorid=targettree["detectorID"].arrays(library="np")["detectorID"]
    elementid=targettree["elementID"].arrays(library="np")["elementID"]
    driftdistance=targettree["driftDistance"].arrays(library="np")["driftDistance"]
    pid=targettree['pid'].arrays(library="np")['pid']
    px=targettree["gpx"].arrays(library="np")['gpx']
    py=targettree["gpy"].arrays(library="np")['gpy']
    pz=targettree["gpz"].arrays(library="np")['gpz']
    vx=targettree["gvx"].arrays(library="np")['gvx']
    vy=targettree["gvy"].arrays(library="np")['gvy']
    vz=targettree["gvz"].arrays(library="np")['gvz']
    nhits=targettree['nhits'].array(library='np')
    for n in range (n_events):
        i=random.randrange(len(detectorid))
        while((nhits[i][0]!=18) or (nhits[i][1]!=18)):i=random.randrange(len(detectorid))
        hits[n],drift[n]=hit_matrix_mc(detectorid[i],elementid[i],driftdistance[i],hits[n],drift[n])
        if(pid[i][0]>0):
            kinematics[n][0]=px[i][0]
            kinematics[n][1]=py[i][0]
            kinematics[n][2]=pz[i][0]
            kinematics[n][3]=px[i][1]
            kinematics[n][4]=py[i][1]
            kinematics[n][5]=pz[i][1]
        if(pid[i][0]<0):
            kinematics[n][0]=px[i][1]
            kinematics[n][1]=py[i][1]
            kinematics[n][2]=pz[i][1]
            kinematics[n][3]=px[i][0]
            kinematics[n][4]=py[i][0]
            kinematics[n][5]=pz[i][0]
        kinematics[n][6]=vx[i][0]
        kinematics[n][7]=vy[i][0]
        kinematics[n][8]=vz[i][0]
    del detectorid, elementid,driftdistance
    
    return hits.astype(bool), drift, kinematics

n_train = 0
train_input = []
val_input = []
train_kinematics = []
val_kinematics = []

while(n_train<1e7):
    valin, valdrift, valkinematics = generate_e906(50000, "Val")
    trainin, traindrift, trainkinematics = generate_e906(500000, "Train")
    
    model = tf.keras.models.load_model('Networks/event_filter')
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    val_predictions = probability_model.predict(valin,batch_size = 256)
    train_predictions = probability_model.predict(trainin,batch_size = 256)

    trainin=trainin[train_predictions[:,3]>0.75]
    traindrift=traindrift[train_predictions[:,3]>0.75]
    train_kinematics.append(trainkinematics[train_predictions[:,3]>0.75])

    valin=valin[val_predictions[:,3]>0.75]
    valdrift=valdrift[val_predictions[:,3]>0.75]
    val_kinematics.append(valkinematics[val_predictions[:,3]>0.75])

    del val_predictions, train_predictions
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model('Networks/Track_Finder_Target')

    predictions = (np.round(model.predict(valin)*max_ele)).astype(int)
    val_input.append(evaluate_finder(valin,valdrift,predictions))
    predictions = (np.round(model.predict(trainin)*max_ele)).astype(int)
    train_input.append(evaluate_finder(trainin,traindrift,predictions))
    n_train+=len(trainin)
    del trainin, valin, traindrift, valdrift, predictions
    print(n_train)
    np.save('Training_Data/Target_Train_In.npy',np.concatenate((train_input)))
    np.save('Training_Data/Target_Val_In.npy',np.concatenate((val_input)))
    np.save('Training_Data/Target_Train_Out.npy',np.concatenate((train_kinematics)))
    np.save('Training_Data/Target_Val_Out.npy',np.concatenate((val_kinematics)))
