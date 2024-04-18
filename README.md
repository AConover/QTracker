# QTracker
Code for QTracker, an ANN-based particle reconstruction system for use with the SeaQuest/SpinQuest spectrometer.

This repository is split into three folders:
1.Monte Carlo Generation Codes
2.Network training
3.Reconstruction code.

************************************************************************************************
The Monte Carlo Generation code is for use with the Fun4All framework maintained by the SpinQuest collaboration. There are 12 scripts used for the current version.
There are four vertex configurations and three mass ranges used for the generation.

The four vertex configurations are:
-All vertices within 25 cm of the beamline from 700 cm upstream of the beam dump to 100 cm inside the beam dump.
-Vertices within 1 cm of the beamline from 700 cm upstream of the beam dump to 100 cm inside the beam dump.
-Vertices within 1 cm of the beamline within the target region (155 cm to 105 cm upstream of the beam dump.)
-Vertices within 1 cm of the beamline within the target dump, from the beginning of the dump to 100 downstream.

The three mass ranges used for the generation are:
-Low-mass region, Drell-Yan events between 2 GeV and 4.5 GeV
-Mid-mass region, Drell-Yan events between 4.5 and 7 GeV
-High-mass region, Drell-Yan events between 7 and 10 GeV.

The generated events are split into training and validation sets. The three mass ranges are then combined for each vertex configuration, and resampled to have a flat mass spectrum between 2 and 9 GeV.

This folder also contains code for generating messy hit matrix events using Monte Carlo for a given process.

************************************************************************************************
The Network training code uses the generated Monte Carlo events to train the sequence of neural networks used for particle reconstruction.
The networks are:
-An event filter that categorizes trigger events by number and charge of detected muons.
-Three track finder networks that identify the detector hits that correspond to the detected muons
-Three momentum reconstruction networks that use the results of the track finder to predict the three-momentum of the detected muons.
-Two vertex finding networks that use the results of the track finder and momentum reconstruction to predict the location that the muons were produced.
-A target-dump filter that uses the results of the track finder, momentum reconstruction, and vertex finding networks to determine the probability that a dimuon pair was produced either in the target region or the dump region of the experiment.

They should be trained in the following order:
1. Event Filter
2. Track Finder Networks (can be trained simultaneously)
3. Momentum Reconstruction (can be trained simultaneously)
4. Vertex Finding (can be trained simultaneously)
5. Target-Dump Filter

Because each network is highly dependent on the output of the previous network, any significant changes to a previous network will necessitate the retraining of downstream networks.

************************************************************************************************
The Reconstruction code has two subdirectories:
1. QTracker_Rivanna: Code to run over a set of SRawEvent digit files. Current configuration allows for reconstruction of E906 events present on Rivanna.
2. QTracker_Run: Code to run for a specified file of events. This code can either reconstruct Monte Carlo events (.npz extension) or experimental events (.root extension, SRawEvent format)











