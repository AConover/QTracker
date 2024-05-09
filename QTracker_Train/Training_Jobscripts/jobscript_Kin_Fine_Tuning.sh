#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH --constraint=a100_80gb
#SBATCH -c 1
#SBATCH -t 72:00:00
#SBATCH -J QTracking_Target_Dump
#SBATCH -o Slurm_Files/QTracking_Tuning.out
#SBATCH -e Slurm_Files/QTracking_Tuning.err
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Python_Files/Generate_Reco_All.py
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Python_Files/Kin_Fine_Tuning.py
