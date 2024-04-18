#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH --constraint=a100_80gb
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J QTracking_Training_TFT
#SBATCH -o QTracking_Training_TFT.out
#SBATCH -e QTracking_TFT.err
#SBATCH --mem=128000

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Track_Finder_Training_Target.py
