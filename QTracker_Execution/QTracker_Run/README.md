This script runs QTracker on a specified file. To execute it on Shannon, use the command:

python3 QTracker_Run.py /path/to/file

It outputs a file in the folder 'Reconstructed' named filename_reconstructed.npy

The rows are as follows:

Row 1: Probability that the event has no reconstructable muons.
Row 2: Probability that the event has one reconstructable muon.
Row 3: Probability that the event has two reconstructable muons of the same sign.
Row 4: Probability that the event has two reconstructable muons of opposite signs (dimuon classification).
Row 5: Probability that the event has three reconstructable muons of the same sign.
Row 6: Probability that the event has three reconstructable muons, two of the same sign, one of the opposite sign.

Rows 7-12 are the px, py, and pz of the positive then negative muons, making no assumptions about vertex position.
Rows 13-5 are the vertex positions of the dimuon pair, making no assumptions about the vertex position.

Rows 16-21 are the px, py, and pz of the positive then negative muons, assuming the dimuon originated along the beamline.
Rows 22-24 are the vertex positions of the dimuon pair, assuming the dimuon originated along the beamline.

Rows 25-30 are the px, py, and pz of the positive then negative muons, assuming the dimuon originated in the target region.

Row 31: Probability that the dimuon pair originated in the dump.
Row 32: Probability that the dimuon pair originated in the target.

Row 33: Run ID
Row 34: Event ID
Row 35: Spill ID
Row 36: Trigger Bit
Row 37: Target Position
Row 38: Turn ID
Row 39: RFID
Rows 40-72: Intensity
Rows 73-76: Number of trigger roads
Rows 77-131: Number of hits per detector before timing cuts.