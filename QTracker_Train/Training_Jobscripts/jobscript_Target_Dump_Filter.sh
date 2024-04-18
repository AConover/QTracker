#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH --constraint=a100_80gb
#SBATCH -c 1
#SBATCH -t 72:00:00
#SBATCH -J QTracking_Target_Dump
#SBATCH -o Slurm_Files/QTracking_Target_Dump.out
#SBATCH -e Slurm_Files/QTracking_Target_Dump.err
#SBATCH --mem=384000

module purge
module load apptainer tensorflow/2.13.0

#apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Python_Files/Generate_Reco_Target.py
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Python_Files/Generate_Reco_Dump.py
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Python_Files/Target_Dump_Filter.py
