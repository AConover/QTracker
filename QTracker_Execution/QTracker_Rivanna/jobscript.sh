#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -c 1
#SBATCH -t 8:00:00
#SBATCH -J QTracking
#SBATCH -o QTracking.out
#SBATCH -e QTracking.err
#SBATCH --mem=384000

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif QTracker_Rivanna.py
