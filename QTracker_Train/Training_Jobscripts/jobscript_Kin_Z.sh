#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:4  
#SBATCH --cpus-per-task=4
#SBATCH -t 72:00:00
#SBATCH -J QTracking_Z_Reco
#SBATCH -o Slurm_Files/QTracking_Z_Reco.out
#SBATCH -e Slurm_Files/QTracking_Z_Reco.err
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Python_Files/Generate_Training_Dimuon.py Z
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Python_Files/Kin_Training_Dimuon.py Z

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Python_Files/Generate_Training_Dimuon.py Z
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Python_Files/Vertex_Training_Dimuon.py Z
