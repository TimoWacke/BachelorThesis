#!/usr/bin/env bash

#SBATCH -J JohannesMeuer
#SBATCH -p amd
#SBATCH -A bb1152
#SBATCH -n 1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --nodelist=vader3

module source start-scripts/setup-modules.txt

singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img_levante.sif \
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/evaluate.py \
 --device cuda --image-size 512 --pooling-layers 3 --encoding-layers 4 --data-type pr \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-complete-scaled/ \
 --mask-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/single_radar_fail.h5 \
 --snapshot-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/radolan-single-radar-fail/ckpt/200000.pth \
 --evaluation-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/evaluation/precipitation/radolan-single-radar-fail/ \
 --lstm-steps 3 \
 --partitions 6027 \
 --create-report \
# --create-images 2017-07-12-14:00,2017-07-12-14:00 \
# --create-video \