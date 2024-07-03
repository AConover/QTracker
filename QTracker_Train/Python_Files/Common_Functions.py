import numpy as np
import uproot
from numba import njit, prange

def read_root_file(root_file):
    targettree = uproot.open(root_file + ':QA_ana')
    targetevents = len(targettree['n_tracks'].array(library='np'))

    mc_data = read_mc_data(targettree)

    pos_events = np.zeros((targetevents, 54))
    pos_drift = np.zeros((targetevents, 30))
    pos_kinematics = np.zeros((targetevents, 6))
    neg_events = np.zeros((targetevents, 54))
    neg_drift = np.zeros((targetevents, 30))
    neg_kinematics = np.zeros((targetevents, 6))
    
    process_target_events(targetevents, mc_data, pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics)
    
    return pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics

def read_mc_data(targettree):
    mc_data = {}
    for keys in ['D0U_ele', 'D0Up_ele', 'D0X_ele', 'D0Xp_ele', 'D0V_ele', 'D0Vp_ele', 
                'D2U_ele', 'D2Up_ele', 'D2X_ele', 'D2Xp_ele', 'D2V_ele', 'D2Vp_ele', 
                'D3pU_ele', 'D3pUp_ele', 'D3pX_ele', 'D3pXp_ele', 'D3pV_ele', 'D3pVp_ele', 
                'D3mU_ele', 'D3mUp_ele', 'D3mX_ele', 'D3mXp_ele', 'D3mV_ele', 'D3mVp_ele', 
                'H1B_ele', 'H1T_ele', 'H1L_ele', 'H1R_ele', 'H2L_ele', 'H2R_ele', 'H2B_ele', 'H2T_ele', 
                'H3B_ele', 'H3T_ele', 'H4Y1L_ele', 'H4Y1R_ele', 'H4Y2L_ele', 'H4Y2R_ele', 'H4B_ele', 'H4T_ele', 
                'P1Y1_ele', 'P1Y2_ele', 'P1X1_ele', 'P1X2_ele', 'P2X1_ele', 'P2X2_ele', 'P2Y1_ele', 'P2Y2_ele', 
                'D0U_drift', 'D0Up_drift', 'D0X_drift', 'D0Xp_drift', 'D0V_drift', 'D0Vp_drift', 
                'D2U_drift', 'D2Up_drift', 'D2X_drift', 'D2Xp_drift', 'D2V_drift', 'D2Vp_drift', 
                'D3pU_drift', 'D3pUp_drift', 'D3pX_drift', 'D3pXp_drift', 'D3pV_drift', 'D3pVp_drift', 
                'D3mU_drift', 'D3mUp_drift', 'D3mX_drift', 'D3mXp_drift', 'D3mV_drift', 'D3mVp_drift',
                'gpx', 'gpy', 'gpz', 'gvx', 'gvy', 'gvz', 'pid']:
        mc_data[keys] = targettree[keys].array(library='np')
    return mc_data

def process_target_events(targetevents, mc_data, pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics):
    pid = mc_data['pid']
    gpx = mc_data['gpx']
    gpy = mc_data['gpy']
    gpz = mc_data['gpz']
    gvx = mc_data['gvx']
    gvy = mc_data['gvy']
    gvz = mc_data['gvz']
    
    for j in range(targetevents):
        first = pid[j][0]
        pos, neg = (0, 1) if first > 0 else (1, 0)
        
        fill_kinematics(gpx, gpy, gpz, gvx, gvy, gvz, pos_kinematics, neg_kinematics, j, pos, neg)
        fill_events_and_drift(mc_data, pos_events, pos_drift, neg_events, neg_drift, j, pos, neg)

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

def fill_events_and_drift(mc_data, pos_events, pos_drift, neg_events, neg_drift, j, pos, neg):
    keys = ['D0U_ele', 'D0Up_ele', 'D0X_ele', 'D0Xp_ele', 'D0V_ele', 'D0Vp_ele', 
                'D2U_ele', 'D2Up_ele', 'D2X_ele', 'D2Xp_ele', 'D2V_ele', 'D2Vp_ele', 
                'D3pU_ele', 'D3pUp_ele', 'D3pX_ele', 'D3pXp_ele', 'D3pV_ele', 'D3pVp_ele', 
                'D3mU_ele', 'D3mUp_ele', 'D3mX_ele', 'D3mXp_ele', 'D3mV_ele', 'D3mVp_ele', 
                'H1B_ele', 'H1T_ele', 'H1L_ele', 'H1R_ele', 'H2L_ele', 'H2R_ele', 'H2B_ele', 'H2T_ele', 
                'H3B_ele', 'H3T_ele', 'H4Y1L_ele', 'H4Y1R_ele', 'H4Y2L_ele', 'H4Y2R_ele', 'H4B_ele', 'H4T_ele', 
                'P1Y1_ele', 'P1Y2_ele', 'P1X1_ele', 'P1X2_ele', 'P2X1_ele', 'P2X2_ele', 'P2Y1_ele', 'P2Y2_ele', 
                'D0U_drift', 'D0Up_drift', 'D0X_drift', 'D0Xp_drift', 'D0V_drift', 'D0Vp_drift', 
                'D2U_drift', 'D2Up_drift', 'D2X_drift', 'D2Xp_drift', 'D2V_drift', 'D2Vp_drift', 
                'D3pU_drift', 'D3pUp_drift', 'D3pX_drift', 'D3pXp_drift', 'D3pV_drift', 'D3pVp_drift', 
                'D3mU_drift', 'D3mUp_drift', 'D3mX_drift', 'D3mXp_drift', 'D3mV_drift', 'D3mVp_drift']
    for k in range(54):
        pos_events[j][k] = mc_data[keys[k]][j][pos]
        neg_events[j][k] = mc_data[keys[k]][j][neg]
    for k in range(30):
        pos_drift[j][k] = mc_data[keys[k+54]][j][pos]
        neg_drift[j][k] = mc_data[keys[k+54]][j][neg]

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


def build_background(n_events):
    filelist=['output_part1.root:tree_nim3','output_part2.root:tree_nim3','output_part3.root:tree_nim3',
             'output_part4.root:tree_nim3','output_part5.root:tree_nim3','output_part6.root:tree_nim3',
             'output_part7.root:tree_nim3','output_part8.root:tree_nim3','output_part9.root:tree_nim3']
    targettree = uproot.open("/project/ptgroup/QTracker_Training/NIM3/"+random.choice(filelist))
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
    return hits, drift

# Function to evaluate the Track Finder neural network.
@njit(parallel=True)
def evaluate_finder(testin, testdrift, predictions):
    # The function constructs inputs for the neural network model based on test data
    # and predictions, processing each event in parallel for efficiency.
    reco_in = np.zeros((len(testin), 68, 3))
    
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
        if(testin[i][j][k - 1]==1):
            reco_in[i][dummy + j_offset][2]=1

    for i in prange(predictions.shape[0]):
        for dummy in prange(34):
            process_entry(i, dummy, 0)
        
        for dummy in prange(34):
            process_entry(i, dummy, 34)      

    return reco_in

# Drift chamber mismatch calculation (function for reusability)
def calc_mismatches(track):
    results = []
    for pos_slice, neg_slice in [(slice(0, 6), slice(34, 40)), (slice(6, 12), slice(40, 46)), (slice(12, 18), slice(46, 52))]:
        results.extend([
            np.sum(abs(track[:, pos_slice, ::2, 0] - track[:, pos_slice, 1::2, 0]) > 1, axis=1),
            np.sum(abs(track[:, neg_slice, ::2, 0] - track[:, neg_slice, 1::2, 0]) > 1, axis=1),
            np.sum(abs(track[:, pos_slice, :, 2]) == 0, axis=1),
            np.sum(abs(track[:, neg_slice, :, 2]) == 0, axis=1)
        ])
    return np.array(results)
