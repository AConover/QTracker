#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH --constraint=a100_80gb
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J QTracking_Target_Reco
#SBATCH -o QTracking_Target_Reco.out
#SBATCH -e QTracking_Target_Reco.err
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Generate_Training_Target.py
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Kin_Training_Target.py
