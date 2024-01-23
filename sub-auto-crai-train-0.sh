#!/usr/bin/env bash

#SBATCH -J crai-train
#SBATCH --output /home/k/k204233/crai/scripts/logs/crai_crai-train_%j.log
#SBATCH -p gpu
#SBATCH -A bb1093
#SBATCH --time=12:00:00
#SBATCH --mem=485G
#SBATCH --exclusive
#SBATCH --constraint a100_80

cd /work/bk1318/k204233/crai/src/climatereconstructionAI/
module load python3
source activate crai

python -m climatereconstructionai.train --load-from-file /work/bk1318/k204233/crai/snapshots/192x92/sr-16x16-bil/run_3/run_3.inp
