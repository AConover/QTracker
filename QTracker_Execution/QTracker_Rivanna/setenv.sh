#!/bin/bash



# Load the Tensorflow Module

module load apptainer tensorflow/2.13.0



# Install Uproot (other libraries are all included in the module)

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif -m pip install --user uproot
