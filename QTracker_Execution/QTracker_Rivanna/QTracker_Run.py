###QTracker Execution###
#This script is used for reconstructing a single root or numpy file.
#Usage:
"""""""""""""""
python QTracker_Run.py /path/to/file.root|.npz
"""""""""""""""
#####Reconstruction Options#####
dimuon_prob_threshold = 0.75 #Minimum dimuon probability to reconstruct.
timing_cuts = True #Use SRawEvent intime flag for hit filtering

#####Output Options#####
event_prob_output = True #Output the event filter probabilites for reconstructed events
track_quality_output = True #Output the number of drift chamber mismatches for each chamber
target_prob_output = True #Output the probability that the dimuon pair is from the target.
tracks_output = False #Output the element IDs for the identified tracks for all three track finders
metadata_output = True #Output metadata

####Metadata Options#####
#Select which values from the SRawEvent file should be saved to the reconstructed .npy file
#Only affects output if using .root file.
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
import uproot  # For reading ROOT files, a common data format in particle physics.
import numba  # Just-In-Time (JIT) compiler for speeding up Python code.
from numba import njit, prange  # njit for compiling functions, prange for parallel loops.
import tensorflow as tf  # For using machine learning models.
import sys

# Check if the script is run without a ROOT file or with the script name as input.
if len(sys.argv) != 2:
    print("Usage: python script_name.py <input_file.root|.npz>")
    quit()

root_file = sys.argv[1]  # Takes the first command-line argument as the input file path.

# Check if the input file has a valid extension
valid_extensions = ('.root', '.npz')
file_extension = os.path.splitext(root_file)[1]
if file_extension not in valid_extensions:
    print("Invalid input file format. Supported formats: .root, .npy")
    quit()

def save_output():
	# After processing through all models, the results are aggregated based on options at top,
	# and the final dataset is prepared.
	
	# The reconstructed kinematics and vertex information are normalized
	# using predefined means and standard deviations before saving.
	row_count = 0
	definitions_string = []	
	if file_extension == '.root':
		metadata = []
		metadata_row_count = 0
		metadata_string = []
		if runid_output:metadata.append(runid)
		if eventid_output:metadata.append(eventid)
		if spillid_output:metadata.append(spill_id)
		if triggerbit_output:metadata.append(trigger_bit)
		if target_pos_output:metadata.append(target_position)
		if turnid_output:metadata.append(turnid)
		if rfid_output:metadata.append(rfid)
		if intensity_output:metadata.append(intensity)
		if trigg_rds_output:metadata.append(n_roads)
		if occ_output:
			if occ_before_cuts:metadata.append(n_hits)
			else:metadata.append(np.sum(hits,axis=2))#Calculates the occupanceis from the Hit Matrix 
		metadata = np.column_stack(metadata)
	if file_extension == '.npz':
		metadata = truth
	output = []
	if event_prob_output: output.append(event_classification_probabilies)
	    output.append(all_predictions)
	if target_prob_output:
	    output.append(target_dump_prob[:,1])
	if track_quality_output:
	    output.append(muon_track_quality)
	    output.append(dimuon_track_quality)
	if tracks_output: output.append(tracks)
	if metadata_output: output.append(metadata)
	output_data = np.column_stack(output)
	
	base_filename = 'Reconstructed/' + os.path.basename(root_file).split('.')[0]
	os.makedirs("Reconstructed", exist_ok=True)  # Ensure the output directory exists.
	np.save(base_filename + '_reconstructed.npy', output_data)  # Save the final dataset.
	print(f"File {base_filename}_reconstructed.npy has been saved successfully.")


# Define normalization constants for kinematic and vertex data.
kin_means = np.array([2, 0, 35, -2, 0, 35])
kin_stds = np.array([0.6, 1.2, 10, 0.6, 1.2, 10])
vertex_means = np.array([0, 0, -300])
vertex_stds = np.array([10, 10, 300])
# Combine kinematic and vertex means and standard deviations for later normalization.
means = np.concatenate((kin_means, vertex_means))
stds = np.concatenate((kin_stds, vertex_stds))

# Function to convert raw detector data into a structured hit matrix.
@njit()
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
        #Add hodoscope hits
        if (detectorid[j]>30) and (detectorid[j]<47) and (intime[j]>0):
            if (tdc[int(detectorid[j])-1][int(elementid[j]-1)]==0) or (tdctime[j]<tdc[int(detectorid[j])-1][int(elementid[j]-1)]):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
                tdc[int(detectorid[j])-1][int(elementid[j]-1)]=tdctime[j]
    return hits,drift,tdc
    
# Predefined maximum element IDs for different detector stations.
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

# Function to remove closely spaced hits that are likely not real particle interactions (cluster hits).
@njit(parallel=True)
def declusterize(hits, drift, tdc):
    # This function iterates over hits and removes clusters of hits that are too close
    # together, likely caused by noise or multiple hits from a single particle passing
    # through the detector. It's an important step in cleaning the data for analysis.
    for k in prange(len(hits)):
        for i in range(54):
            if(i<30 or i>45):
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

def calc_mismatches(track):
    results = []
    for pos_slice, neg_slice in [(slice(0, 6), slice(34, 40)), (slice(6, 12), slice(40, 46)), (slice(12, 18), slice(46, 52))]:
        # Compare even indices with odd indices for the 0th component of the final dimension
        even_pos_indices = track[:, pos_slice, 0].reshape(track.shape[0], -1, 2)[:, :, 0]
        odd_pos_indices = track[:, pos_slice, 0].reshape(track.shape[0], -1, 2)[:, :, 1]
        even_neg_indices = track[:, neg_slice, 0].reshape(track.shape[0], -1, 2)[:, :, 0]
        odd_neg_indices = track[:, neg_slice, 0].reshape(track.shape[0], -1, 2)[:, :, 1]

        results.extend([
            np.sum(abs(even_pos_indices - odd_pos_indices) > 1, axis=1),
            np.sum(abs(even_neg_indices - odd_neg_indices) > 1, axis=1),
            np.sum(track[:, pos_slice, 2] == 0, axis=1),
            np.sum(track[:, neg_slice, 2] == 0, axis=1)
        ])
    
    return np.array(results)


if(file_extension=='.root'):
	# Read in data from the ROOT file.
	targettree = uproot.open(root_file + ":save")
	# The ROOT file contains structured data from particle detection events, including
	# identifiers for each hit in the detectors, the time of each hit, and the measured
	# drift time, which is related to the distance of the particle from the detector element.

	#Event Data
	detectorid = targettree["fAllHits.detectorID"].arrays(library="np")["fAllHits.detectorID"]
	elementid = targettree["fAllHits.elementID"].arrays(library="np")["fAllHits.elementID"]
	driftdistance = targettree["fAllHits.driftDistance"].arrays(library="np")["fAllHits.driftDistance"]
	tdctime = targettree["fAllHits.tdcTime"].arrays(library="np")["fAllHits.tdcTime"]
	intime = targettree["fAllHits.flag"].arrays(library="np")["fAllHits.flag"]

	# Initialize arrays to hold processed hit data.
	hits = np.zeros((len(detectorid), 54, 201),dtype=bool)
	drift = np.zeros((len(detectorid), 54, 201))
	tdc = np.zeros((len(detectorid), 54, 201),dtype=int)

	# Process each event to fill the hits, drift, and TDC arrays with cleaned and structured data.
	for n in range(len(detectorid)):
	    hits[n], drift[n], tdc[n] = hit_matrix(detectorid[n], elementid[n], driftdistance[n],tdctime[n],intime[n], hits[n], drift[n], tdc[n])

	declusterize(hits, drift, tdc)  # Remove closely spaced hits.

	tf.keras.backend.clear_session()  # Clear any existing TensorFlow sessions.
	tf.compat.v1.reset_default_graph()  # Reset the default graph to prepare for model loading.

if(file_extension=='.npz'):
	generated = np.load(root_file)
	
	# Extract the arrays from the loaded data
	hits = generated["hits"]
	drift = generated["drift"]
	truth = generated["truth"]

print("Loaded events")

# Load and apply a pre-trained TensorFlow model for event filtering.
model = tf.keras.models.load_model('Networks/event_filter')
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
event_classification_probabilies  = probability_model.predict(hits, batch_size=256, verbose=0)

# Filter out events based on the prediction from the event filter model.
#Keep events that have better than 75% probability of having a dimuon tracks.
filt = event_classification_probabilies [:, 1] > dimuon_prob_threshold
hits = hits[filt]
drift = drift[filt]

tf.keras.backend.clear_session()
model = tf.keras.models.load_model('../Networks/Track_Finder_Pos')
pos_predictions = model.predict(hits, verbose=0)
tf.keras.backend.clear_session()
model = tf.keras.models.load_model('../Networks/Track_Finder_Neg')
neg_predictions =model.predict(hits, verbose=0)
predictions = (np.round(np.column_stack((pos_predictions,neg_predictions))*max_ele)).astype(int)

muon_track=evaluate_finder(hits,drift,predictions)

tf.keras.backend.clear_session()
model = tf.keras.models.load_model('../Networks/Reconstruction_Pos')
pos_pred = model.predict(muon_track[:,:34,:2], verbose=0)
tf.keras.backend.clear_session()
model = tf.keras.models.load_model('../Networks/Reconstruction_Neg')
neg_pred = model.predict(muon_track[:,34:,:2], verbose=0)

muon_track_quality = calc_mismatches(muon_track)
filt1 = ((muon_track_quality[0::4] < 2) & (muon_track_quality[1::4] < 2) & (muon_track_quality[2::4] < 3) & (muon_track_quality[3::4] < 3)).all(axis=0)

hits = hits[filt1]
drift = drift[filt1]

if(file_extension=='.root'):
	# Read and filter metadata based on the same criteria used for hits and drift data.
	# This metadata includes various identifiers and measurements related to the events.
	runid = targettree["fRunID"].arrays(library="np")["fRunID"][filt][filt1]
        eventid = targettree["fEventID"].arrays(library="np")["fEventID"][filt][filt1]
        spill_id = targettree["fSpillID"].arrays(library="np")["fSpillID"][filt][filt1]
        trigger_bit = targettree["fTriggerBits"].arrays(library="np")["fTriggerBits"][filt][filt1]
        target_position = targettree["fTargetPos"].arrays(library="np")["fTargetPos"][filt][filt1]
        turnid = targettree["fTurnID"].arrays(library="np")["fTurnID"][filt][filt1]
        rfid = targettree["fRFID"].arrays(library="np")["fRFID"][filt][filt1]
        intensity = targettree["fIntensity[33]"].arrays(library="np")["fIntensity[33]"][filt][filt1]
        n_roads = targettree["fNRoads[4]"].arrays(library="np")["fNRoads[4]"][filt][filt1]
        n_hits = targettree["fNHits[55]"].arrays(library="np")["fNHits[55]"][filt][filt1]

if(file_extension=='.npz'):
	truth = truth[filt][filt1]

event_classification_probabilies  = event_classification_probabilies [filt][filt1]  # Apply the filter to the predictions as well.
 pos_pred = pos_pred[filt1]
neg_pred = neg_pred[filt1]
muon_track_quality = muon_track_quality.T[filt1]

print("Filtered Events")
print("Found",len(hits),"dimuons.")

# If there are filtered events to process, continue with the track finding and reconstruction.
if(len(hits>0)):
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    model = tf.keras.models.load_model('../Networks/Track_Finder_All')
    predictions = (np.round(model.predict(hits,verbose=0)*max_ele)).astype(int)
    all_vtx_track = evaluate_finder(hits,drift,predictions)[:,:,:2]

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
    z_vtx_track = evaluate_finder(hits,drift,predictions)[:,:,:2]

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
    target_vtx_reco = model.predict(target_track[:,:,:2],batch_size=8192,verbose=0)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    model = tf.keras.models.load_model('../Networks/Track_Finder_Dump')
    predictions = (np.round(model.predict(hits,verbose=0)*max_ele)).astype(int)
    dump_track = evaluate_finder(hits,drift,predictions)[:,:,:2]

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    model=tf.keras.models.load_model('../Networks/Reconstruction_Dump')
    dump_vtx_reco = model.predict(dump_track,batch_size=8192,verbose=0)

    dimuon_track_quality = calc_mismatches(target_track).T

    reco_kinematics = np.concatenate((event_classification_probabilies[:,1], pos_pred, neg_pred, all_vtx_reco, z_vtx_reco, target_vtx_reco, dump_vtx_reco, muon_track_quality, dimuon_track_quality),axis=1)

    tracks = np.column_stack((muon_track[:,:,:2], all_vtx_track, z_vtx_track, target_track[:,:,:2], dump_track))

    target_dump_input = np.column_stack((reco_kinematics,tracks.reshape((len(tracks),(68*2*5)))))

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    model=tf.keras.models.load_model('../Networks/target_dump_filter')
    target_dump_pred = model.predict(target_dump_input,batch_size=512,verbose=0)
    target_dump_prob = np.exp(target_dump_pred) / np.sum(np.exp(target_dump_pred), axis=1, keepdims=True)

    all_predictions = np.column_stack((pos_pred, neg_pred, all_vtx_reco*stds+means, z_vtx_reco*stds+means, target_vtx_reco*kin_stds+kin_means, dump_vtx_reco*kin_stds+kin_means))            
    
    print("Found ",len(all_predictions)," Dimuons in file.")
    
    save_output()
    print("QTracker Complete")
    

else:
    print("No events meeting dimuon criteria.")  # If no events pass the filter, notify the user.
