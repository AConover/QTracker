###QTracker Rivanna###
# This script is used to reconstruct large amount of data on Rivanna via Slurm job submission.

#####Parent path Directory#####
root_directory = '/project/ptgroup/seaquest/data/digit/02/'

#####Reconstruction Options#####
dimuon_prob_threshold = 0.75 #Minimum dimuon probability to reconstruct.
timing_cuts = True #Use SRawEvent intime flag for hit filtering

#####Output Options#####
event_prob_output = True #Output the event filter probabilites for reconstructed events
n_mismatch_output = True #Output the number of drift chamber mismatches for each chamber
tracks_output = False #Output the element IDs for the identified tracks for all three track finders
metadata_output = True #Output metadata

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

def save_explanation():
    explanation = []
    n_columns = 0
    if event_prob_output: 
        explanation.append(f"Event Filter Probabilites: Columns {n_columns}:{n_columns+6}")
        n_columns+=6
    if n_mismatch_output:
        explanation.append(f"Drift chamber mismatches (St 1, 2, 3): Columns {n_columns}:{n_columns+3}")
        n_columns+=3
    explanation.append(f"All Vertex Kinematic Predictions: Columns {n_columns}:{n_columns+6}")
    n_columns+=6
    explanation.append(f"All Vertex Vertex Predictions: Columns {n_columns}:{n_columns+3}")
    n_columns+=3
    explanation.append(f"Z Vertex Kinematic Predictions: Columns {n_columns}:{n_columns+6}")
    n_columns+=6
    explanation.append(f"Z Vertex Vertex Predictions: Columns {n_columns}:{n_columns+3}")
    n_columns+=3
    explanation.append(f"Target Vertex Kinematic Predictions: Columns {n_columns}:{n_columns+6}")
    n_columns+=6
    if target_prob_output:
        explanation.append(f"Target Probability: Column {n_columns}")
        n_columns+=1
    if tracks_output: 
        explanation.append(f"All Vertex Track: {n_columns}:{n_columns+68}")
        n_columns+=68
        explanation.append(f"Z Vertex Track: {n_columns}:{n_columns+68}")
        n_columns+=68
        explanation.append(f"Target Vertex Track: {n_columns}:{n_columns+68}")
        n_columns+=68
    if metadata_output & (file_extension == '.root'):
        if runid_output:
            explanation.append(f"Run ID: Column {n_columns}")
            n_columns+=1
        if eventid_output:
            explanation.append(f"Event ID: Column {n_columns}")
            n_columns+=1
        if spillid_output:
            explanation.append(f"Spill ID: Column {n_columns}")
            n_columns+=1
        if triggerbit_output:
            explanation.append(f"Trigger Bits: Column {n_columns}")
            n_columns+=1
        if target_pos_output:
            explanation.append(f"Target Positions: Column {n_columns}")
            n_columns+=1
        if turnid_output:
            explanation.append(f"Turn ID: Column {n_columns}")
            n_columns+=1
        if rfid_output:
            explanation.append(f"RFID: Column {n_columns}")
            n_columns+=1
        if intensity_output:
            explanation.append(f"Cherenkov Information: Columns {n_columns}:{n_columns+32}")
            n_columns+=32
        if trigg_rds_output:
            explanation.append(f"Number of Trigger Roads: Column {n_columns}")
            n_columns+=1
        if occ_output:
            if occ_before_cuts:explanation.append(f"Detector Occupancies before cuts: Columns {n_columns}:{n_columns+54}")
            else:explanation.append(f"Detector Occupancies after cuts: Columns {n_columns}:{n_columns+54}")
            n_columns+=54
    if (file_extension == '.npz'):
        explanation.append(f"Truth Kinematics: Columns {n_columns}:{n_columns+6}")
        n_columns+=6
        explanation.append(f"Truth Vertex: Columns {n_columns}:{n_columns+3}")
    basename = os.path.basename(root_file).split('.')[0]
    filename= f'Reconstructed/{basename}_columns.txt'
    os.makedirs("Reconstructed", exist_ok=True)  # Ensure the output directory exists.
    with open(filename,'w') as file:
        file.write('Explanation of Columns:\n\n')
        for info in explanation:
            file.write(f"{info}\n")    

save_explanation()            

def save_output():
    # After processing through all models, the results are aggregated based on options at top,
    # and the final dataset is prepared.

    # The reconstructed kinematics and vertex information are normalized
    # using predefined means and standard deviations before saving.
    output = []
    if event_prob_output: output.append(event_classification_probabilies)
    if n_mismatch_output:
        output.append(dc_unmatched_st_1)
        output.append(dc_unmatched_st_2)
        output.append(dc_unmatched_st_3)
    output.append(all_predictions)
    if target_prob_output:
        output.append(target_dump_prob[:,1])
    if tracks_output: output.append(tracks)
    if runid_output:output.append(runid)
    if eventid_output:output.append(eventid)
    if spillid_output:output.append(spill_id)
    if triggerbit_output:output.append(trigger_bit)
    if target_pos_output:output.append(target_position)
    if turnid_output:output.append(turnid)
    if rfid_output:output.append(rfid)
    if intensity_output:output.append(intensity)
    if trigg_rds_output:output.append(n_roads)
    if occ_output:
        if occ_before_cuts:output.append(n_hits)
        else:output.append(np.sum(hits,axis=2))#Calculates the occupanceis from the Hit Matrix 
    metadata = np.column_stack(metadata)
    output_data = np.column_stack(output)

    base_filename = 'Reconstructed/' + os.path.basename(root_file).split('.')[0]
    os.makedirs("Reconstructed", exist_ok=True)  # Ensure the output directory exists.
    np.save(base_filename + '_reconstructed.npy', output_data)  # Save the final dataset.
    print(f"File {base_filename}_reconstructed.npy has been saved successfully.")

        
kin_means = np.array([ 2.00, 0.00, 35.0, -2.00, -0.00, 35.0 ])
kin_stds = np.array([ 0.6, 1.2, 10.00, 0.60, 1.20, 10.00 ])

vertex_means=np.array([0,0,-300])
vertex_stds=np.array([10,10,300])

means = np.concatenate((kin_means,vertex_means))
stds = np.concatenate((kin_stds,vertex_stds))

@njit(nopython=True)
def hit_matrix(detectorid,elementid,drifttime,tdctime,intime,hits,drift,tdc): #Convert into hit matrices
    if(timing_cuts==False):intime[:]=2
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
        #Apply hodoscope timing cuts
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
    def set_to_zero(k, i, j):
        hits[k][i][j] = 0
        drift[k][i][j] = 0
        tdc[k][i][j] = 0
    
    for k in prange(len(hits)):
        for i in range(31):
            for j in range(100):#Work from both sides
                if(hits[k][i][j]==1 and hits[k][i][j+1]==1):
                    if(hits[k][i][j+2]==0):#Two hits
                        #Edge Hit Check
                        if(drift[k][i][j]>0.4 and drift[k][i][j+1]>0.9):set_to_zero(k,i,j+1)
                        elif(drift[k][i][j+1]>0.4 and drift[k][i][j]>0.9):set_to_zero(k,i,j)
                        #Electronic Noise Check
                        if(abs(tdc[k][i][j]-tdc[k][i][j+1])<8):
                            set_to_zero(k,i,j)
                            set_to_zero(k,i,j+1)
                    else:#Check larger clusters for Electronic Noise
                        n=2
                        while(hits[k][i][j+n]==1):n=n+1
                        dt_mean = 0
                        for m in range(n-1):
                            dt_mean += (tdc[k][i][j+m]-tdc[k][i][j+m+1])
                        dt_mean = dt_mean/(n-1)
                        if(dt_mean<10):
                            for m in range(n):
                                set_to_zero(k,i,j+m)
                                
                if(hits[k][i][200-j]==1 and hits[k][i][199-j]):
                    if(hits[k][i][198-j]==0):
                        #Edge Hit Check
                        if(drift[k][i][200-j]>0.4 and drift[k][i][199-j]>0.9):set_to_zero(k,i,199-j)  # Edge hit check
                        elif(drift[k][i][199-j]>0.4 and drift[k][i][200-j]>0.9):set_to_zero(k,i,200-j)  # Edge hit check
                        # Electronic Noise Check
                        if(abs(tdc[k][i][200-j]-tdc[k][i][199-j])<8):
                            set_to_zero(k,i,200-j)
                            set_to_zero(k,i,199-j)
                    else:  # Check larger clusters for Electronic Noise
                        n=2
                        while(hits[k][i][200-j-n]==1): n=n+1
                        dt_mean = 0
                        for m in range(n-1):
                            dt_mean += abs(tdc[k][i][200-j-m]-tdc[k][i][200-j-m-1])
                        dt_mean = dt_mean/(n-1)
                        if(dt_mean<10):
                            for m in range(n):
                                set_to_zero(k,i,200-j-m)                           


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
        event_classification_probabilies = probability_model.predict(hits,batch_size=256)
        
        filt = event_classification_probabilies[:,3]>0.75
        
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


        event_classification_probabilies = event_classification_probabilies[filt]
        
        print("Filtered Events")
        if(len(hits>0)):
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model = tf.keras.models.load_model('../Networks/Track_Finder_All')
            predictions = (np.round(model.predict(hits,verbose=0)*max_ele)).astype(int)
            all_vtx_track = evaluate_finder(hits,drift,predictions)

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model=tf.keras.models.load_model('../Networks/Reconstruction_All')
            reco_kinematics = model.predict(all_vtx_track,batch_size=8192,verbose=0)

            vertex_input=np.concatenate((reco_kinematics.reshape((len(reco_kinematics),3,2)),all_vtx_track),axis=1)

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model=tf.keras.models.load_model('../Networks/Vertexing_All')
            reco_vertex = model.predict(vertex_input,batch_size=8192,verbose=0)

            all_vtx_reco=np.concatenate((reco_kinematics,reco_vertex),axis=1)

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model = tf.keras.models.load_model('../Networks/Track_Finder_Z')
            predictions = (np.round(model.predict(hits,verbose=0)*max_ele)).astype(int)
            z_vtx_track = evaluate_finder(hits,drift,predictions)

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model=tf.keras.models.load_model('../Networks/Reconstruction_Z')
            reco_kinematics = model.predict(z_vtx_track,batch_size=8192,verbose=0)
            
            vertex_input=np.concatenate((reco_kinematics.reshape((len(reco_kinematics),3,2)),z_vtx_track),axis=1)

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model=tf.keras.models.load_model('../Networks/Vertexing_Z')
            reco_vertex = model.predict(vertex_input,batch_size=8192,verbose=0)

            z_vtx_reco=np.concatenate((reco_kinematics,reco_vertex),axis=1)

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model = tf.keras.models.load_model('../Networks/Track_Finder_Target')
            predictions = (np.round(model.predict(hits,verbose=0)*max_ele)).astype(int)
            target_track = evaluate_finder(hits,drift,predictions)

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model=tf.keras.models.load_model('../Networks/Reconstruction_Target')
            reco_kinematics = model.predict(target_track,batch_size=8192,verbose=0)
            print("Reconstructed events for target vertices")

            reco_kinematics = np.concatenate((all_vtx_reco,z_vtx_reco,target_vtx_reco_kinematics),axis=1)
            
            tracks = np.column_stack((all_vtx_track, z_vtx_track, target_track))
            
            target_dump_input = np.column_stack((reco_kinematics,tracks.reshape((len(tracks),(204*2)))))
            
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model=tf.keras.models.load_model('../Networks/target_dump_filter')
            target_dump_pred = model.predict(target_dump_input,batch_size=512,verbose=0)
            target_dump_prob = np.exp(target_dump_pred) / np.sum(np.exp(target_dump_pred), axis=1, keepdims=True)
            
            #Calculate the number of drift chamber mismatches for output
            st1_track = np.column_stack((target_track[:,:6,0],target_track[:,34:40,0]))
            st2_track = np.column_stack((target_track[:,6:12,0],target_track[:,40:46,0]))
            st3_track = np.column_stack((target_track[:,12:18,0],target_track[:,46:52,0]))
            dc_unmatched_st_1 = np.sum(abs(st1_track[:,::2]-st1_track[:,1::2])>1,axis=1)
            dc_unmatched_st_2 = np.sum(abs(st2_track[:,::2]-st2_track[:,1::2])>1,axis=1)
            dc_unmatched_st_3 = np.sum(abs(st3_track[:,::2]-st3_track[:,1::2])>1,axis=1)
            
            all_predictions = np.column_stack((all_vtx_reco*stds+means,z_vtx_reco*stds+means, target_vtx_reco_kinematics*kin_stds+kin_means))            
            
            save_output()
        else: print("No events meeting dimuon criteria.")
    except Exception as e:
        # Handle the exception (print an error message, log it, etc.)
        print("Error processing file {root_file}: ",e)
        continue

    

