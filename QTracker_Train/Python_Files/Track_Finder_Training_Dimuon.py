import tensorflow as tf
import numpy as np
from numba import njit, prange
import random
import gc
import sys
from Python_Files.Common_Functions import *

if len(sys.argv) != 2:
    print("Usage: python script.py <Vertex Distribution>")
    print("Currently supports All, Z, Target, and Dump")
    exit(1)

vertex = sys.argv[1]
root_file_train = f"Root_Files/{vertex}_Train_QA_v2.root"
root_file_val = f"Root_Files/{vertex}_Val_QA_v2.root"
model_name = f"Networks/Track_Finder_{vertex}"

pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics = read_root_file(root_file_train)
pos_events_val, pos_drift_val, pos_kinematics_val, neg_events_val, neg_drift_val, neg_kinematics_val = read_root_file(root_file_val)

del pos_drift, neg_drift, pos_kinematics, neg_kinematics
del pos_drift_val, neg_drift_val, pos_kinematics_val, neg_kinematics_val

pos_events = clean(pos_events).astype(int)
neg_events = clean(neg_events).astype(int)
pos_events_val = clean(pos_events_val).astype(int)
neg_events_val = clean(neg_events_val).astype(int)

@njit(parallel=True)
def track_injection(hits, pos_e, neg_e):
    n_events = len(hits)
    track_real = np.zeros((n_events, 68), dtype=np.float32) 

    for z in prange(n_events):
        j = np.random.randint(len(pos_e))
        for k in range(54):
            pos_val = pos_e[j][k]
            neg_val = neg_e[j][k]
            if pos_val > 0 and (np.random.random() < 0.94 or k > 29):
                hits[z][k][int(pos_val - 1)] = 1
            if neg_val > 0 and (np.random.random() < 0.94 or k > 29):
                hits[z][k][int(neg_val - 1)] = 1

        # Convert the hits into tracks to be reconstructed.
        track_real[z, :6] = pos_e[j, :6]  
        track_real[z, 6:12] = pos_e[j, 12:18]
        track_real[z, 34:40] = neg_e[j, :6]  
        track_real[z, 40:46] = neg_e[j, 12:18]
        # St. 3p gets positive values, St. 3m gets negative values.
        track_real[z, 12:18] = np.where((pos_e[j, 18]) > 0, pos_e[j, 18:24], -pos_e[j, 24:30])
        track_real[z, 46:52] = np.where(neg_e[j, 18] > 0, neg_e[j, 18:24], -neg_e[j, 24:30])
        # Pairs of hodoscopes are mutually exclusive, this gives positive or negative values depending on the array.
        track_real[z, 18:26] = np.where(pos_e[j, 30:45:2] > 0, pos_e[j, 30:45:2], -pos_e[j, 31:46:2])
        track_real[z, 52:60] = np.where(neg_e[j, 30:45:2] > 0 , neg_e[j, 30:45:2], -neg_e[j, 31:46:2])
        track_real[z, 26:34] = pos_e[j, 46:54]
        track_real[z, 60:68] = neg_e[j, 46:54]

    return hits, track_real

def generate_hit_matrices(n_events, tvt):
    # Create the realistic background for events
    hits, _ = build_background(n_events)
    # Place the full tracks that are reconstructable
    if(tvt == "Train"):
        hits, track = track_injection(hits, pos_events, neg_events)    
    if(tvt == "Val"):
        hits, track = track_injection(hits, pos_events_val, neg_events_val)    
    return hits.astype(bool), track.astype(int)
    
learning_rate_finder = 1e-5
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
n_train = 0

# Detect the number of GPUs available
gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
print(f"Number of GPUs available: {num_gpus}")

# Set up strategy for distributed training
if num_gpus > 1:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()

# Adjust batch size for the number of GPUs
batch_size_ef = 256 * num_gpus
batch_size_tf = 64 * num_gpus    

print("Before while loop:", n_train)
while(n_train < 1e7):
    trainin, traintrack = generate_hit_matrices(750000, "Train")
    print("Generated Training Data")
    traintrack = traintrack / max_ele
    
    valin, valtrack = generate_hit_matrices(75000, "Val")
    print("Generated Validation Data")
    valtrack = valtrack / max_ele
  
    # Clear previous session and load probability model
    tf.keras.backend.clear_session()

    with strategy.scope():
        probability_model = tf.keras.Sequential([tf.keras.models.load_model('Networks/event_filter'), tf.keras.layers.Softmax()])
        train_predictions = probability_model.predict(trainin, batch_size=batch_size_ef, verbose=0)
        val_predictions = probability_model.predict(valin, batch_size=batch_size_ef, verbose=0)

    train_mask = train_predictions[:, 1] > 0.75
    val_mask = val_predictions[:, 1] > 0.75

    # Clear session and load track finder models then predict with track finders using strategy
    tf.keras.backend.clear_session()
    
    with strategy.scope():
        track_finder_pos = tf.keras.models.load_model('Networks/Track_Finder_Pos')
        pos_predictions_val = (np.round(track_finder_pos.predict(valin, verbose=0, batch_size = batch_size_tf) * max_ele)).astype(int)
        pos_predictions_train = (np.round(track_finder_pos.predict(trainin, verbose=0, batch_size = batch_size_tf) * max_ele)).astype(int)
    
    tf.keras.backend.clear_session()
    
    with strategy.scope():
        track_finder_neg = tf.keras.models.load_model('Networks/Track_Finder_Neg')
        neg_predictions_val = (np.round(track_finder_neg.predict(valin, verbose=0) * max_ele)).astype(int)
        neg_predictions_train = (np.round(track_finder_neg.predict(trainin, verbose=0) * max_ele)).astype(int)
    
    track_val = evaluate_finder(valin, valdrift, np.column_stack((pos_predictions_val, neg_predictions_val)))
    results_val = calc_mismatches(track_val)
    val_mask &= ((results_val[0::4] < 2) & (results_val[1::4] < 2) & (results_val[2::4] < 3) & (results_val[3::4] < 3)).all(axis=0)
    
    track_train = evaluate_finder(trainin, traindrift, np.column_stack((pos_predictions_train, neg_predictions_train)))
    results_train = calc_mismatches(track_train)
    train_mask &= ((results_train[0::4] < 2) & (results_train[1::4] < 2) & (results_train[2::4] < 3) & (results_train[3::4] < 3)).all(axis=0)

    # Apply masks
    trainin = trainin[train_mask]
    traintrack = traintrack[train_mask]
    valin = valin[val_mask]
    valtrack = valtrack[val_mask]

    n_train += len(trainin)
        
    # Model Training
    tf.keras.backend.clear_session()

    with strategy.scope():
        model = tf.keras.models.load_model(model_name)
        optimizer = tf.keras.optimizers.Adam(learning_rate_finder)
        model.compile(optimizer=optimizer, loss='mse', metrics=['RootMeanSquaredError'])
        val_loss_before = model.evaluate(valin, valtrack, batch_size=batch_size_tf, verbose=2)[0]
        print(val_loss_before)
        history = model.fit(trainin, traintrack, epochs=1000, batch_size=batch_size_tf, 
                            verbose=2, validation_data=(valin, valtrack), callbacks=[callback])
        if min(history.history['val_loss']) < val_loss_before:
            model.save(model_name)  # Save only if improved
            learning_rate_finder *= 2
        else:
            learning_rate_finder /= 2

    del model  # Delete the model to free up memory
    gc.collect()  # Force garbage collection to release GPU memory
    print(n_train)
