#!/usr/bin/env bash

#SBATCH -J JohannesMeuer
#SBATCH -p gpu
#SBATCH -A bb1152
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --mem=64G
###SBATCH --nodelist=mg206

module load cuda/10.0.130
module load singularity/3.6.1-gcc-9.1.0
export HDF5_USE_FILE_LOCKING='FALSE'
singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img.sif bash /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/start.sh