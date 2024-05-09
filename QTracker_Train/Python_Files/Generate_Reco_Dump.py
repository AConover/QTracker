import os
import numpy as np
import uproot
from numba import njit, prange
import random
import tensorflow as tf
import gc

def read_root_file(root_file):
    print("Reading ROOT files...")
    targettree = uproot.open(root_file+':QA_ana')
    targetevents=len(targettree['n_tracks'].array(library='np'))
    D0U_ele = targettree['D0U_ele'].array(library='np')
    D0Up_ele = targettree['D0Up_ele'].array(library='np')
    D0X_ele = targettree['D0X_ele'].array(library='np')
    D0Xp_ele = targettree['D0Xp_ele'].array(library='np')
    D0V_ele = targettree['D0V_ele'].array(library='np')
    D0Vp_ele = targettree['D0Vp_ele'].array(library='np')

    D2U_ele = targettree['D2U_ele'].array(library='np')
    D2Up_ele = targettree['D2Up_ele'].array(library='np')
    D2X_ele = targettree['D2X_ele'].array(library='np')
    D2Xp_ele = targettree['D2Xp_ele'].array(library='np')
    D2V_ele = targettree['D2V_ele'].array(library='np')
    D2Vp_ele = targettree['D2Vp_ele'].array(library='np')

    D3pU_ele = targettree['D3pU_ele'].array(library='np')
    D3pUp_ele = targettree['D3pUp_ele'].array(library='np')
    D3pX_ele = targettree['D3pX_ele'].array(library='np')
    D3pXp_ele = targettree['D3pXp_ele'].array(library='np')
    D3pV_ele = targettree['D3pV_ele'].array(library='np')
    D3pVp_ele = targettree['D3pVp_ele'].array(library='np')

    D3mU_ele = targettree['D3mU_ele'].array(library='np')
    D3mUp_ele = targettree['D3mUp_ele'].array(library='np')
    D3mX_ele = targettree['D3mX_ele'].array(library='np')
    D3mXp_ele = targettree['D3mXp_ele'].array(library='np')
    D3mV_ele = targettree['D3mV_ele'].array(library='np')
    D3mVp_ele = targettree['D3mVp_ele'].array(library='np')

    D0U_drift = targettree['D0U_drift'].array(library='np')
    D0Up_drift = targettree['D0Up_drift'].array(library='np')
    D0X_drift = targettree['D0X_drift'].array(library='np')
    D0Xp_drift = targettree['D0Xp_drift'].array(library='np')
    D0V_drift = targettree['D0V_drift'].array(library='np')
    D0Vp_drift = targettree['D0Vp_drift'].array(library='np')

    D2U_drift = targettree['D2U_drift'].array(library='np')
    D2Up_drift = targettree['D2Up_drift'].array(library='np')
    D2X_drift = targettree['D2X_drift'].array(library='np')
    D2Xp_drift = targettree['D2Xp_drift'].array(library='np')
    D2V_drift = targettree['D2V_drift'].array(library='np')
    D2Vp_drift = targettree['D2Vp_drift'].array(library='np')

    D3pU_drift = targettree['D3pU_drift'].array(library='np')
    D3pUp_drift = targettree['D3pUp_drift'].array(library='np')
    D3pX_drift = targettree['D3pX_drift'].array(library='np')
    D3pXp_drift = targettree['D3pXp_drift'].array(library='np')
    D3pV_drift = targettree['D3pV_drift'].array(library='np')
    D3pVp_drift = targettree['D3pVp_drift'].array(library='np')

    D3mU_drift = targettree['D3mU_drift'].array(library='np')
    D3mUp_drift = targettree['D3mUp_drift'].array(library='np')
    D3mX_drift = targettree['D3mX_drift'].array(library='np')
    D3mXp_drift = targettree['D3mXp_drift'].array(library='np')
    D3mV_drift = targettree['D3mV_drift'].array(library='np')
    D3mVp_drift = targettree['D3mVp_drift'].array(library='np')

    H1B_ele = targettree['H1B_ele'].array(library='np')
    H1T_ele = targettree['H1T_ele'].array(library='np')
    H1L_ele = targettree['H1L_ele'].array(library='np')
    H1R_ele = targettree['H1R_ele'].array(library='np')

    H2L_ele = targettree['H2L_ele'].array(library='np')
    H2R_ele = targettree['H2R_ele'].array(library='np')
    H2B_ele = targettree['H2B_ele'].array(library='np')
    H2T_ele = targettree['H2T_ele'].array(library='np')

    H3B_ele = targettree['H3B_ele'].array(library='np')
    H3T_ele = targettree['H3T_ele'].array(library='np')

    H4Y1L_ele = targettree['H4Y1L_ele'].array(library='np')
    H4Y1R_ele = targettree['H4Y1R_ele'].array(library='np')
    H4Y2L_ele = targettree['H4Y2L_ele'].array(library='np')
    H4Y2R_ele = targettree['H4Y2R_ele'].array(library='np')
    H4B_ele = targettree['H4B_ele'].array(library='np')
    H4T_ele = targettree['H4T_ele'].array(library='np')

    P1Y1_ele = targettree['P1Y1_ele'].array(library='np')
    P1Y2_ele = targettree['P1Y2_ele'].array(library='np')
    P1X1_ele = targettree['P1X1_ele'].array(library='np')
    P1X2_ele = targettree['P1X2_ele'].array(library='np')

    P2X1_ele = targettree['P2X1_ele'].array(library='np')
    P2X2_ele = targettree['P2X2_ele'].array(library='np')
    P2Y1_ele = targettree['P2Y1_ele'].array(library='np')
    P2Y2_ele = targettree['P2Y2_ele'].array(library='np')

    gpx = targettree['gpx'].array(library='np')
    gpy = targettree['gpy'].array(library='np')
    gpz = targettree['gpz'].array(library='np')
    gvx = targettree['gvx'].array(library='np')
    gvy = targettree['gvy'].array(library='np')
    gvz = targettree['gvz'].array(library='np')

    pid = targettree['pid'].array(library='np')

    print('Done')

    #This reads the dimuon tracks from the target into an array
    pos_events=np.zeros((targetevents,54))
    pos_drift = np.zeros((targetevents,30))
    pos_kinematics = np.zeros((targetevents,6))
    neg_events=np.zeros((targetevents,54))
    neg_drift = np.zeros((targetevents,30))
    neg_kinematics = np.zeros((targetevents,6))
    print("Reading target events...")
    for j in range(targetevents):
        first=pid[j][0]
        if(first>0):
            pos=0
            neg=1
        else:
            pos=1
            neg=0
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
        pos_events[j][0]=D0U_ele[j][pos]
        neg_events[j][0]=D0U_ele[j][neg]
        pos_events[j][1]=D0Up_ele[j][pos]
        neg_events[j][1]=D0Up_ele[j][neg]
        pos_events[j][2]=D0X_ele[j][pos]
        neg_events[j][2]=D0X_ele[j][neg]        
        pos_events[j][3]=D0Xp_ele[j][pos]
        neg_events[j][3]=D0Xp_ele[j][neg]
        pos_events[j][4]=D0V_ele[j][pos]
        neg_events[j][4]=D0V_ele[j][neg]
        pos_events[j][5]=D0Vp_ele[j][pos]
        neg_events[j][5]=D0Vp_ele[j][neg]
        pos_events[j][16]=D2U_ele[j][pos]
        neg_events[j][16]=D2U_ele[j][neg]
        pos_events[j][17]=D2Up_ele[j][pos]
        neg_events[j][17]=D2Up_ele[j][neg]
        pos_events[j][15]=D2X_ele[j][pos]
        neg_events[j][15]=D2X_ele[j][neg]
        pos_events[j][14]=D2Xp_ele[j][pos]
        neg_events[j][14]=D2Xp_ele[j][neg]
        pos_events[j][12]=D2V_ele[j][pos]
        neg_events[j][12]=D2V_ele[j][neg]
        pos_events[j][13]=D2Vp_ele[j][pos]
        neg_events[j][13]=D2Vp_ele[j][neg]
        pos_events[j][23]=D3pU_ele[j][pos]
        neg_events[j][23]=D3pU_ele[j][neg]
        pos_events[j][22]=D3pUp_ele[j][pos]
        neg_events[j][22]=D3pUp_ele[j][neg]
        pos_events[j][21]=D3pX_ele[j][pos]
        neg_events[j][21]=D3pX_ele[j][neg]
        pos_events[j][20]=D3pXp_ele[j][pos]
        neg_events[j][20]=D3pXp_ele[j][neg]
        pos_events[j][19]=D3pV_ele[j][pos]
        neg_events[j][19]=D3pV_ele[j][neg]
        pos_events[j][18]=D3pVp_ele[j][pos]
        neg_events[j][18]=D3pVp_ele[j][neg]
        pos_events[j][29]=D3mU_ele[j][pos]
        neg_events[j][29]=D3mU_ele[j][neg]
        pos_events[j][28]=D3mUp_ele[j][pos]
        neg_events[j][28]=D3mUp_ele[j][neg]
        pos_events[j][27]=D3mX_ele[j][pos]
        neg_events[j][27]=D3mX_ele[j][neg]
        pos_events[j][26]=D3mXp_ele[j][pos]
        neg_events[j][26]=D3mXp_ele[j][neg]
        pos_events[j][25]=D3mV_ele[j][pos]
        neg_events[j][25]=D3mV_ele[j][neg]
        pos_events[j][24]=D3mVp_ele[j][pos]
        neg_events[j][24]=D3mVp_ele[j][neg]
        pos_events[j][30]=H1B_ele[j][pos]
        neg_events[j][30]=H1B_ele[j][neg]
        pos_events[j][31]=H1T_ele[j][pos]
        neg_events[j][31]=H1T_ele[j][neg]
        pos_events[j][32]=H1L_ele[j][pos]
        neg_events[j][32]=H1L_ele[j][neg]
        pos_events[j][33]=H1R_ele[j][pos]
        neg_events[j][33]=H1R_ele[j][neg]
        pos_events[j][34]=H2L_ele[j][pos]
        neg_events[j][34]=H2L_ele[j][neg]
        pos_events[j][35]=H2R_ele[j][pos]
        neg_events[j][35]=H2R_ele[j][neg]
        pos_events[j][36]=H2T_ele[j][pos]
        neg_events[j][36]=H2T_ele[j][neg]
        pos_events[j][37]=H2B_ele[j][pos]
        neg_events[j][37]=H2B_ele[j][neg]
        pos_events[j][38]=H3B_ele[j][pos]
        neg_events[j][38]=H3B_ele[j][neg]
        pos_events[j][39]=H3T_ele[j][pos]
        neg_events[j][39]=H3T_ele[j][neg]
        pos_events[j][40]=H4Y1L_ele[j][pos]
        neg_events[j][40]=H4Y1L_ele[j][neg]
        pos_events[j][41]=H4Y1R_ele[j][pos]
        neg_events[j][41]=H4Y1R_ele[j][neg]
        pos_events[j][42]=H4Y2L_ele[j][pos]
        neg_events[j][42]=H4Y2L_ele[j][neg]
        pos_events[j][43]=H4Y2R_ele[j][pos]
        neg_events[j][43]=H4Y2R_ele[j][neg]
        pos_events[j][44]=H4B_ele[j][pos]
        neg_events[j][44]=H4B_ele[j][neg]
        pos_events[j][45]=H4T_ele[j][pos]
        neg_events[j][45]=H4T_ele[j][neg]
        pos_events[j][46]=P1Y1_ele[j][pos]
        neg_events[j][46]=P1Y1_ele[j][neg]
        pos_events[j][47]=P1Y2_ele[j][pos]
        neg_events[j][47]=P1Y2_ele[j][neg]
        pos_events[j][48]=P1X1_ele[j][pos]
        neg_events[j][48]=P1X1_ele[j][neg]
        pos_events[j][49]=P1X2_ele[j][pos]
        neg_events[j][49]=P1X2_ele[j][neg]
        pos_events[j][50]=P2X1_ele[j][pos]
        neg_events[j][50]=P2X1_ele[j][neg]
        pos_events[j][51]=P2X2_ele[j][pos]
        neg_events[j][51]=P2X2_ele[j][neg]
        pos_events[j][52]=P2Y1_ele[j][pos]
        neg_events[j][52]=P2Y1_ele[j][neg]
        pos_events[j][53]=P2Y2_ele[j][pos]
        neg_events[j][53]=P2Y2_ele[j][neg]
        pos_drift[j][pos]=D0U_drift[j][pos]
        pos_drift[j][pos]=D0U_drift[j][pos]
        neg_drift[j][pos]=D0U_drift[j][neg]
        pos_drift[j][neg]=D0Up_drift[j][pos]
        neg_drift[j][neg]=D0Up_drift[j][neg]
        pos_drift[j][2]=D0X_drift[j][pos]
        neg_drift[j][2]=D0X_drift[j][neg]        
        pos_drift[j][3]=D0Xp_drift[j][pos]
        neg_drift[j][3]=D0Xp_drift[j][neg]
        pos_drift[j][4]=D0V_drift[j][pos]
        neg_drift[j][4]=D0V_drift[j][neg]
        pos_drift[j][5]=D0Vp_drift[j][pos]
        neg_drift[j][5]=D0Vp_drift[j][neg]
        pos_drift[j][16]=D2U_drift[j][pos]
        neg_drift[j][16]=D2U_drift[j][neg]
        pos_drift[j][17]=D2Up_drift[j][pos]
        neg_drift[j][17]=D2Up_drift[j][neg]
        pos_drift[j][15]=D2X_drift[j][pos]
        neg_drift[j][15]=D2X_drift[j][neg]
        pos_drift[j][14]=D2Xp_drift[j][pos]
        neg_drift[j][14]=D2Xp_drift[j][neg]
        pos_drift[j][12]=D2V_drift[j][pos]
        neg_drift[j][12]=D2V_drift[j][neg]
        pos_drift[j][13]=D2Vp_drift[j][pos]
        neg_drift[j][13]=D2Vp_drift[j][neg]
        pos_drift[j][23]=D3pU_drift[j][pos]
        neg_drift[j][23]=D3pU_drift[j][neg]
        pos_drift[j][22]=D3pUp_drift[j][pos]
        neg_drift[j][22]=D3pUp_drift[j][neg]
        pos_drift[j][21]=D3pX_drift[j][pos]
        neg_drift[j][21]=D3pX_drift[j][neg]
        pos_drift[j][20]=D3pXp_drift[j][pos]
        neg_drift[j][20]=D3pXp_drift[j][neg]
        pos_drift[j][19]=D3pV_drift[j][pos]
        neg_drift[j][19]=D3pV_drift[j][neg]
        pos_drift[j][18]=D3pVp_drift[j][pos]
        neg_drift[j][18]=D3pVp_drift[j][neg]
        pos_drift[j][29]=D3mU_drift[j][pos]
        neg_drift[j][29]=D3mU_drift[j][neg]
        pos_drift[j][28]=D3mUp_drift[j][pos]
        neg_drift[j][28]=D3mUp_drift[j][neg]
        pos_drift[j][27]=D3mX_drift[j][pos]
        neg_drift[j][27]=D3mX_drift[j][neg]
        pos_drift[j][26]=D3mXp_drift[j][pos]
        neg_drift[j][26]=D3mXp_drift[j][neg]
        pos_drift[j][25]=D3mV_drift[j][pos]
        neg_drift[j][25]=D3mV_drift[j][neg]
        pos_drift[j][24]=D3mVp_drift[j][pos]
        neg_drift[j][24]=D3mVp_drift[j][neg]
    print("Done")

    return pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics


pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics = read_root_file('Root_Files/Dump_Train_QA_v2.root')
pos_events_val, pos_drift_val, pos_kinematics_val, neg_events_val, neg_drift_val, neg_kinematics_val = read_root_file('Root_Files/Dump_Val_QA_v2.root')

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
def track_injection(hits,drift,pos_e,neg_e,pos_d,neg_d,pos_k,neg_k):
    #Start generating the events
    kin=np.zeros((len(hits),9))
    for z in prange(len(hits)):
        j=random.randrange(len(pos_e))
        kin[z][0]=pos_k[j][0]
        kin[z][1]=pos_k[j][1]
        kin[z][2]=pos_k[j][2]
        kin[z][3]=neg_k[j][0]
        kin[z][4]=neg_k[j][1]
        kin[z][5]=neg_k[j][2]
        kin[z][6]=pos_k[j][3]
        kin[z][7]=pos_k[j][4]
        kin[z][8]=pos_k[j][5]
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

@njit()
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

def generate_e906(n_events, tvt):
    #Import NIM3 events and put them on the hit matrices.
    #Randomly choosing between 1 and 6 events per station gets approximately the correct occupancies.
    filelist=['output_part1.root:tree_nim3','output_part2.root:tree_nim3','output_part3.root:tree_nim3',
             'output_part4.root:tree_nim3','output_part5.root:tree_nim3','output_part6.root:tree_nim3',
             'output_part7.root:tree_nim3','output_part8.root:tree_nim3','output_part9.root:tree_nim3']
    targettree = uproot.open("NIM3/"+random.choice(filelist))
    detectorid_nim3=targettree["det_id"].arrays(library="np")["det_id"]
    elementid_nim3=targettree["ele_id"].arrays(library="np")["ele_id"]
    driftdistance_nim3=targettree["drift_dist"].arrays(library="np")["drift_dist"]
    hits = np.zeros((n_events,54,201))
    drift = np.zeros((n_events,54,201))
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

# Initialize lists to store the data
dimuon_probability=[]
all_predictions = []
tracks = []
truth = []
total_entries = 0



while(total_entries<5000000):
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
            dump_vtx_track = evaluate_finder(hits,drift,predictions)            
            print("Found Tracks")

            tf.keras.backend.clear_session()
            
            model=tf.keras.models.load_model('Networks/Reconstruction_Target')
            reco_kinematics = model.predict(dump_vtx_track,batch_size=8192,verbose=0)

            dump_vtx_reco_kinematics= reco_kinematics

            reco_kinematics = np.concatenate((all_vtx_reco_kinematics,z_vtx_reco_kinematics,dump_vtx_reco_kinematics),axis=1)
            
            dimuon_probability.append(dimu_pred)
            all_predictions.append(reco_kinematics)
            truth.append(kinematics)
            tracks.append(np.column_stack((all_vtx_track,z_vtx_track,dump_vtx_track)))
            
            np.save('Training_Data/dump_tracks_train.npy',np.concatenate(tracks, axis=0))
            np.save('Training_Data/dump_kinematics_train.npy',np.concatenate(all_predictions, axis=0))
            np.save('Training_Data/dump_truth_train.npy',np.concatenate(truth, axis=0))
            
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



while(total_entries<500000):
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
            dump_vtx_track = evaluate_finder(hits,drift,predictions)            
            print("Found Tracks")

            tf.keras.backend.clear_session()
            
            model=tf.keras.models.load_model('Networks/Reconstruction_Target')
            reco_kinematics = model.predict(dump_vtx_track,batch_size=8192,verbose=0)

            dump_vtx_reco_kinematics= reco_kinematics

            reco_kinematics = np.concatenate((all_vtx_reco_kinematics,z_vtx_reco_kinematics,dump_vtx_reco_kinematics),axis=1)
            
            dimuon_probability.append(dimu_pred)
            all_predictions.append(reco_kinematics)
            truth.append(kinematics)
            tracks.append(np.column_stack((all_vtx_track,z_vtx_track,dump_vtx_track)))
            
            np.save('Training_Data/dump_tracks_val.npy',np.concatenate(tracks, axis=0))
            np.save('Training_Data/dump_kinematics_val.npy',np.concatenate(all_predictions, axis=0))
            np.save('Training_Data/dump_truth_val.npy',np.concatenate(truth, axis=0))
            
            print("Reconstructed events for target vertices")
            total_entries += len(hits)
            print(total_entries)
            del hits, drift, reco_kinematics, reco_vertex
        else: print("No events meeting dimuon criteria.")
    except Exception as e:
        pass
