#!/usr/bin/env bash

#SBATCH -J crai-train
#SBATCH --output /home/k/k203179/reconstructing-ai-for-weather-station-data/station_reconstruct/slurm_logs/crai_crai-train_%j.log
#SBATCH -p gpu
#SBATCH -A bm1159
#SBATCH --time=12:00:00
#SBATCH --mem=485G
#SBATCH --exclusive
#SBATCH --constraint a100_80

cd /home/k/k203179/reconstructing-ai-for-weather-station-data/
module load python3

# Initialize Conda (add this line)
eval "$(conda shell.bash hook)"

conda activate /home/k/k203179/.conda/envs/crai

python -m climatereconstructionai.train --load-from-file /home/k/k203179/reconstructing-ai-for-weather-station-data/station_reconstruct/train_args_soltau.txt
