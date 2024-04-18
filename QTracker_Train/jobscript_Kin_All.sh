#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH --constraint=a100_80gb
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J QTracking_All_Reco
#SBATCH -o QTracking_All_Reco.out
#SBATCH -e QTracking_All_Reco.err
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.13.0

#apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Generate_Training_All.py
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Kin_Training_All.py

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Generate_Training_All.py
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Vertex_Training_All.py
