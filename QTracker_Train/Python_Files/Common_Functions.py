import numpy
import uproot

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


def build_background():
