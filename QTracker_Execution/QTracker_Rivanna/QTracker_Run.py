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

# Function to remove closely spaced hits that are likely not real particle interactions (cluster hits).
@njit(parallel=True)
def declusterize(hits, drift, tdc):
    # This function iterates over hits and removes clusters of hits that are too close
    # together, likely caused by noise or multiple hits from a single particle passing
    # through the detector. It's an important step in cleaning the data for analysis.
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
predictions = probability_model.predict(hits, batch_size=256, verbose=0)

# Filter out events based on the prediction from the event filter model.
#Keep events that have better than 75% probability of having a dimuon tracks.
filt = predictions[:, 3] > 0.75
hits = hits[filt]
drift = drift[filt]
if(file_extension=='.root'):
	# Read and filter metadata based on the same criteria used for hits and drift data.
	# This metadata includes various identifiers and measurements related to the events.
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

	metadata = np.column_stack((runid, eventid, spill_id, trigger_bit, target_position, turnid, rfid, intensity, n_roads, n_hits))

if(file_extension=='.npz'):
	truth = truth[filt]

predictions = predictions[filt]  # Apply the filter to the predictions as well.

print("Filtered Events")
print("Found",len(hits),"dimuons.")

# If there are filtered events to process, continue with the track finding and reconstruction.
if(len(hits) > 0):
    # The predictions from the event filter are stored for later use.
    dimuon_probability = predictions

    # Clear any existing TensorFlow models from memory to load new ones.
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    # Load the Track Finder model trained to identify tracks across all vertex positions.
    model = tf.keras.models.load_model('Networks/Track_Finder_All')
    predictions = (np.round(model.predict(hits, verbose=0) * max_ele)).astype(int)
    
    # Evaluate the Track Finder model and adjust the hit matrices accordingly.
    all_vtx_track = evaluate_finder(hits, drift, predictions)
    print("Found Tracks")
       
    # Similar steps are repeated for models trained on different subsets of the data,
    # such as those specific to z-vertices or target vertices, to improve the precision
    # of track identification under different conditions.

    # After finding tracks, the next step is to reconstruct the 4-momentum
    # for the particles involved in each event. This involves loading a new model
    # specifically trained for this purpose and processing the track data through it.

    # Clear TensorFlow sessions again to ensure a clean slate for loading new models.
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    # Load the momentum reconstruction model and predict the 4-momentum for each track.
    model = tf.keras.models.load_model('Networks/Reconstruction_All')
    pred = model.predict(all_vtx_track, batch_size=8192, verbose=0)
    reco_kinematics = pred  # Store the predicted 4-momentum for each event.

    # Vertex reconstruction is similar to the previous steps, using a dedicated model
    # to determine the points in space where the particle interactions occurred.

    # Combine the reconstructed kinematic data with the original hit data for vertexing.
    vertex_reco = np.concatenate((pred.reshape((len(pred), 3, 2)), all_vtx_track), axis=1)

    # Clear TensorFlow sessions and load the vertex reconstruction model.
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    model = tf.keras.models.load_model('Networks/Vertexing_All')
    
    # Predict vertex positions for each event.
    pred = model.predict(vertex_reco, batch_size=8192, verbose=0)
    reco_vertex = pred  # Store the reconstructed vertex positions.

    # Combine all reconstructed data for a comprehensive analysis.
    all_vtx_reco_kinematics = np.concatenate((reco_kinematics, reco_vertex), axis=1)

    print("Reconstructed events for all vertices")
 
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    model = tf.keras.models.load_model('Networks/Track_Finder_Z')
    predictions = (np.round(model.predict(hits,verbose=0)*max_ele)).astype(int)
    z_vtx_track = evaluate_finder(hits,drift,predictions)

    # Similar processes are repeated for the tracks identified by
    # such as those specific to z vertices or target vertices.
    # Three versions of the track finder were trained on different vertex distributions:
    # All vertices along the beamline within 1 meter of the beam.
    # All z-vertices along the beamline and finally Target vertices.
    # This multi-model approach allows for a nuanced analysis of particle tracks
    # from various perspectives, improving the overall quality of the reconstruction.

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    model=tf.keras.models.load_model('Networks/Reconstruction_Z')
    pred = model.predict(z_vtx_track,batch_size=8192,verbose=0)
    reco_kinematics = pred

    vertex_reco=np.concatenate((pred.reshape((len(pred),3,2)),z_vtx_track),axis=1)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    model=tf.keras.models.load_model('Networks/Vertexing_Z')
    pred = model.predict(vertex_reco,batch_size=8192,verbose=0)
    reco_vertex = pred

    z_vtx_reco_kinematics=np.concatenate((reco_kinematics,reco_vertex),axis=1)

    print("Reconstructed events for z vertices")
 
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    model = tf.keras.models.load_model('Networks/Track_Finder_Target')
    predictions = (np.round(model.predict(hits,verbose=0)*max_ele)).astype(int)
    target_track = evaluate_finder(hits,drift,predictions)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    model=tf.keras.models.load_model('Networks/Reconstruction_Target')
    pred = model.predict(target_track,batch_size=8192,verbose=0)
    reco_kinematics = pred

    target_vtx_reco_kinematics= reco_kinematics

    reco_kinematics = np.concatenate((all_vtx_reco_kinematics,z_vtx_reco_kinematics,target_vtx_reco_kinematics),axis=1)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    model=tf.keras.models.load_model('Networks/target_dump_filter')
    target_dump_prob = model.predict(reco_kinematics,batch_size=8192,verbose=0)
    tracks = np.concatenate((all_vtx_track, z_vtx_track, target_track),axis=2)
    all_predictions = np.column_stack((all_vtx_reco_kinematics*stds+means,z_vtx_reco_kinematics*stds+means, target_vtx_reco_kinematics*kin_stds+kin_means))            

    print("Reconstructed events for target vertices")
    if(file_extension=='.root'):
    	output_data = np.column_stack((dimuon_probability, all_predictions, target_dump_prob, metadata))
    if(file_extension=='.npz'):
    	output_data = np.column_stack((dimuon_probability, all_predictions, target_dump_prob, truth))
    # After processing through all models, the results are aggregated,
    # and the final dataset is prepared by combining the dimuon probability,
    # reconstructed kinematics, and vertex information with the original event metadata.

    # The reconstructed kinematics and vertex information are normalized
    # using predefined means and standard deviations before saving.

    # The QTracker output data is saved to a NumPy file for further analysis,

    base_filename = 'Reconstructed/' + os.path.basename(root_file).split('.')[0]
    os.makedirs("Reconstructed", exist_ok=True)  # Ensure the output directory exists.
    np.save(base_filename + '_reconstructed.npy', output_data)  # Save the final dataset.
    
    print(f"File {base_filename}_reconstructed.npy has been saved successfully.")
    print("QTracker Complete")
    

else:
    print("No events meeting dimuon criteria.")  # If no events pass the filter, notify the user.
