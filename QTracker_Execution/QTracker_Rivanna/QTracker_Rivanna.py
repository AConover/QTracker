###QTracker Execution###
#This script is used for reconstructing a large number of 

#####Parent Directory ######
#Set the top level directory where SRawEvent files are stored.
root_directory = '/project/ptgroup/seaquest/data/digit/02/'

#####Reconstruction Options#####
dimuon_prob_threshold = 0.75 #Minimum dimuon probability to reconstruct.
timing_cuts = True #Use SRawEvent intime flag for hit filtering

#####Output Options#####
event_prob_output = True #Output the event filter probabilites for reconstructed events
n_mismatch_output = True #Output the number of drift chamber mismatches for each chamber

####Metadata Options#####
#Select which values from the SRawEvent file should be saved to the reconstructed .npy file
runid_output = True #Output the run id
eventid_output = True #Output the event id
spillid_output = True #Output the spill id
triggerbit_output = True #Output the trigger bit for the event
target_pos_output = True #Output the target type (hydrogen, deuterium, etc.)
turnid_output = True #Output the turn id
rfid_output = True #Output the RF ID
intensity_output = True #Output Cherenkov information
trigg_rds_output = True #Output the number of trigger roads activated
occ_output = True #Output the occupancy information
occ_before_cuts = False #If set to true, counts number of hits before timing cuts, if false, outputs occupancies after hit reduction.

#################
import os
import numpy as np
import uproot
import numba
from numba import njit, prange
import tensorflow as tf

kin_means = np.array([ 2.00, 0.00, 35.0, -2.00, -0.00, 35.0 ])
kin_stds = np.array([ 0.6, 1.2, 10.00, 0.60, 1.20, 10.00 ])

vertex_means=np.array([0,0,-300])
vertex_stds=np.array([10,10,300])

means = np.concatenate((kin_means,vertex_means))
stds = np.concatenate((kin_stds,vertex_stds))

@njit(nopython=True)
def hit_matrix(detectorid,elementid,drifttime,tdctime,intime,hits,drift,tdc): #Convert into hit matrices
    for j in prange (len(detectorid)):
        #Apply station 0 TDC timing cuts
        if (detectorid[j]<7) and (intime[j]>0):
            if (tdc[int(detectorid[j])-1][int(elementid[j]-1)]==0) or (tdctime[j]<tdc[int(detectorid[j])-1][int(elementid[j]-1)]):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
                tdc[int(detectorid[j])-1][int(elementid[j]-1)]=tdctime[j]
        #Apply station 2 TDC timing cuts
        if (detectorid[j]>12) and (detectorid[j]<19) and (intime[j]>0):
            if (tdc[int(detectorid[j])-1][int(elementid[j]-1)]==0) or (tdctime[j]<tdc[int(detectorid[j])-1][int(elementid[j]-1)]):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
                tdc[int(detectorid[j])-1][int(elementid[j]-1)]=tdctime[j]
        #Apply station 3p TDC timing cuts
        if (detectorid[j]>18) and (detectorid[j]<25) and (intime[j]>0):
            if (tdc[int(detectorid[j])-1][int(elementid[j]-1)]==0) or (tdctime[j]<tdc[int(detectorid[j])-1][int(elementid[j]-1)]):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
                tdc[int(detectorid[j])-1][int(elementid[j]-1)]=tdctime[j]
        #Apply station 3p TDC timing cuts
        if (detectorid[j]>24) and (detectorid[j]<31) and (intime[j]>0):
            if (tdc[int(detectorid[j])-1][int(elementid[j]-1)]==0) or (tdctime[j]<tdc[int(detectorid[j])-1][int(elementid[j]-1)]):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
                tdc[int(detectorid[j])-1][int(elementid[j]-1)]=tdctime[j]
        #Apply prop tube TDC timing cuts
        if (detectorid[j]>46) and (detectorid[j]<55) and (intime[j]>0):
            if (tdc[int(detectorid[j])-1][int(elementid[j]-1)]==0) or (tdctime[j]<tdc[int(detectorid[j])-1][int(elementid[j]-1)]):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
                tdc[int(detectorid[j])-1][int(elementid[j]-1)]=tdctime[j]
        #Add hodoscope hits
        if (detectorid[j]>30) and (detectorid[j]<47) and (intime[j]>0):
            if (tdc[int(detectorid[j])-1][int(elementid[j]-1)]==0) or (tdctime[j]<tdc[int(detectorid[j])-1][int(elementid[j]-1)]):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
                tdc[int(detectorid[j])-1][int(elementid[j]-1)]=tdctime[j]
    return hits,drift,tdc

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

@njit(parallel=True)
def declusterize(hits,drift,tdc):
    for k in prange(len(hits)):
        for i in range(31):
            for j in range(100):#Work from both sides
                if(hits[k][i][j]==1 and hits[k][i][j+1]==1):
                    if(hits[k][i][j+2]==0):#Two hits
                        if(drift[k][i][j]>0.4 and drift[k][i][j+1]>0.9):#Edge hit check
                            hits[k][i][j+1]=0
                            drift[k][i][j+1]=0
                            tdc[k][i][j+1]=0
                        elif(drift[k][i][j+1]>0.4 and drift[k][i][j]>0.9):#Edge hit check
                            hits[k][i][j]=0
                            drift[k][i][j]=0
                            tdc[k][i][j]=0
                        if(abs(tdc[k][i][j]-tdc[k][i][j+1])<8):#Electronic Noise Check
                            hits[k][i][j+1]=0
                            drift[k][i][j+1]=0
                            tdc[k][i][j+1]=0
                            hits[k][i][j]=0
                            drift[k][i][j]=0
                            tdc[k][i][j]=0
                            
                    else:#Check larger clusters for Electronic Noise
                        n=2
                        while(hits[k][i][j+n]==1):n=n+1
                        dt_mean = 0
                        for m in range(n-1):
                            dt_mean += (tdc[k][i][j+m]-tdc[k][i][j+m+1])
                        dt_mean = dt_mean/(n-1)
                        if(dt_mean<10):
                            for m in range(n):
                                hits[k][i][j+m]=0
                                drift[k][i][j+m]=0
                                tdc[k][i][j+m]=0
                if(hits[k][i][200-j]==1 and hits[k][i][199-j]):
                    if(hits[k][i][198-j]==0):
                        if(drift[k][i][200-j]>0.4 and drift[k][i][199-j]>0.9):  # Edge hit check
                            hits[k][i][199-j]=0
                            drift[k][i][199-j]=0
                        elif(drift[k][i][199-j]>0.4 and drift[k][i][200-j]>0.9):  # Edge hit check
                            hits[k][i][200-j]=0
                            drift[k][i][200-j]=0
                        if(abs(tdc[k][i][200-j]-tdc[k][i][199-j])<8):  # Electronic Noise Check
                            hits[k][i][199-j]=0
                            drift[k][i][199-j]=0
                            tdc[k][i][199-j]=0
                            hits[k][i][200-j]=0
                            drift[k][i][200-j]=0
                            tdc[k][i][200-j]=0
                    else:  # Check larger clusters for Electronic Noise
                        n=2
                        while(hits[k][i][200-j-n]==1): n=n+1
                        dt_mean = 0
                        for m in range(n-1):
                            dt_mean += abs(tdc[k][i][200-j-m]-tdc[k][i][200-j-m-1])
                        dt_mean = dt_mean/(n-1)
                        if(dt_mean<10):
                            for m in range(n):
                                hits[k][i][200-j-m]=0
                                drift[k][i][200-j-m]=0
                                tdc[k][i][200-j-m]=0                               


# Specify the directory containing the root files
i = 0

# List all root files in the directory
root_files = [file for file in os.listdir(root_directory) if file.endswith('.root')]

for root_file in root_files[i:]:
    try:
        # Open the current root file
        i += 1
        print(i)
        print(root_file)
        targettree = uproot.open(os.path.join(root_directory, root_file) + ":save")

        #Event Data
        detectorid = targettree["fAllHits.detectorID"].arrays(library="np")["fAllHits.detectorID"]
        elementid = targettree["fAllHits.elementID"].arrays(library="np")["fAllHits.elementID"]
        driftdistance = targettree["fAllHits.driftDistance"].arrays(library="np")["fAllHits.driftDistance"]
        tdctime = targettree["fAllHits.tdcTime"].arrays(library="np")["fAllHits.tdcTime"]
        intime = targettree["fAllHits.flag"].arrays(library="np")["fAllHits.flag"]

        hits = np.zeros((len(detectorid), 54, 201),dtype=bool)
        drift = np.zeros((len(detectorid), 54, 201))
        tdc = np.zeros((len(detectorid), 54, 201),dtype=int)

        for n in range(len(detectorid)):
            hits[n], drift[n], tdc[n] = hit_matrix(detectorid[n], elementid[n], driftdistance[n],tdctime[n],intime[n], hits[n], drift[n], tdc[n])
        
        declusterize(hits, drift, tdc)
        
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        print("Loaded events")
        model = tf.keras.models.load_model('../Networks/event_filter')
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(hits,batch_size=256)
        
        filt = predictions[:,3]>0.75
        
        hits=hits[filt]
        drift=drift[filt]

		#Meta Data
        runid = targettree["fRunID"].arrays(library="np")["fRunID"][filt]
        eventid = targettree["fEventID"].arrays(library="np")["fEventID"][filt]
        spill_id = targettree["fSpillID"].arrays(library="np")["fSpillID"][filt]
        trigger_bit = targettree["fTriggerBits"].arrays(library="np")["fTriggerBits"][filt]
        target_position = targettree["fTargetPos"].arrays(library="np")["fTargetPos"][filt]
        turnid = targettree["fTurnID"].arrays(library="np")["fTurnID"][filt]
        rfid = targettree["fRFID"].arrays(library="np")["fRFID"][filt]
        intensity = targettree["fIntensity[33]"].arrays(library="np")["fIntensity[33]"][filt]
        n_roads = targettree["fNRoads[4]"].arrays(library="np")["fNRoads[4]"][filt]
        n_hits = targettree["fNHits[55]"].arrays(library="np")["fNHits[55]"][filt]
        
        metadata=np.column_stack((runid, eventid, spill_id, trigger_bit, target_position, turnid, rfid,
                                         intensity, n_roads, n_hits))       


        predictions = predictions[filt]
        
        print("Filtered Events")
        if(len(hits>0)):
            dimuon_probability=predictions
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model = tf.keras.models.load_model('../Networks/Track_Finder_All')
            predictions = (np.round(model.predict(hits)*max_ele)).astype(int)
            all_vtx_track = evaluate_finder(hits,drift,predictions)
            print("Found Tracks")


            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model=tf.keras.models.load_model('../Networks/Reconstruction_All')
            pred = model.predict(all_vtx_track,batch_size=8192)
            reco_kinematics = pred

            vertex_reco=np.concatenate((pred.reshape((len(pred),3,2)),all_vtx_track),axis=1)

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model=tf.keras.models.load_model('../Networks/Vertexing_All')
            pred = model.predict(vertex_reco,batch_size=8192)
            reco_vertex = pred

            all_vtx_reco_kinematics=np.concatenate((reco_kinematics,reco_vertex),axis=1)

            print("Reconstructed events for all vertices")

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model = tf.keras.models.load_model('../Networks/Track_Finder_Z')
            predictions = (np.round(model.predict(hits)*max_ele)).astype(int)
            z_vtx_track = evaluate_finder(hits,drift,predictions)
            print("Found Tracks")


            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model=tf.keras.models.load_model('../Networks/Reconstruction_Z')
            pred = model.predict(z_vtx_track,batch_size=8192)
            reco_kinematics = pred

            vertex_reco=np.concatenate((pred.reshape((len(pred),3,2)),z_vtx_track),axis=1)

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model=tf.keras.models.load_model('../Networks/Vertexing_Z')
            pred = model.predict(vertex_reco,batch_size=8192)
            reco_vertex = pred

            z_vtx_reco_kinematics=np.concatenate((reco_kinematics,reco_vertex),axis=1)

            print("Reconstructed events for z vertices")

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model = tf.keras.models.load_model('../Networks/Track_Finder_Target')
            predictions = (np.round(model.predict(hits)*max_ele)).astype(int)
            target_track = evaluate_finder(hits,drift,predictions)
            print("Found Tracks")
		
            dc_track = np.column_stack((target_track[:,:18,0],target_track[:,34:52,0]))
            dc_unmatched_count = np.sum(abs(dc_track[:,::2]-dc_track[:,1::2])>1,axis=1)


            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model=tf.keras.models.load_model('../Networks/Reconstruction_Target')
            pred = model.predict(target_track,batch_size=8192)
            reco_kinematics = pred

            target_vtx_reco_kinematics= reco_kinematics

            reco_kinematics = np.concatenate((all_vtx_reco_kinematics,z_vtx_reco_kinematics,target_vtx_reco_kinematics),axis=1)
            
            tracks = np.column_stack((all_vtx_track, z_vtx_track, target_track))
            
            target_dump_input = np.column_stack((reco_kinematics,tracks.reshape((len(tracks),(204*2)))))
            
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model=tf.keras.models.load_model('../Networks/target_dump_filter')
            target_dump_pred = model.predict(target_dump_input,batch_size=512)
            target_dump_prob = np.exp(target_dump_pred) / np.sum(np.exp(target_dump_pred), axis=1, keepdims=True)
            all_predictions = np.column_stack((all_vtx_reco_kinematics*stds+means,z_vtx_reco_kinematics*stds+means, target_vtx_reco_kinematics*kin_stds+kin_means))            
            print("Reconstructed events for target vertices")

            output_data = np.column_stack((dimuon_probability,dc_unmatched_count, all_predictions, target_dump_prob, metadata))

            base_filename = 'Reconstructed/'+os.path.basename(root_file).split('.')[0]
            os.makedirs("Reconstructed", exist_ok=True)
            np.save(base_filename+'_reconstructed.npy', output_data)
            del hits, drift, pred, reco_kinematics, reco_vertex
        else: print("No events meeting dimuon criteria.")
    except Exception as e:
        # Handle the exception (print an error message, log it, etc.)
        print("Error processing file {root_file}: ",e)
        continue

    

