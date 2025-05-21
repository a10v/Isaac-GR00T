#!/bin/bash
#SBATCH --account=p32775                      ## e.g. p30157
#SBATCH --partition=gengpu                    ## GPU partition (48 h max)
#SBATCH --gres=gpu:a100:1                     ## 1 A100 GPU
#SBATCH --nodes=1                             ## single node
#SBATCH --mem=48G                             ## RAM
#SBATCH --time=12:00:00                       ## wall-time
#SBATCH --job-name=gr00t_finetune             ## job name
#SBATCH --output=gr00t_finetune_%j.out        ## stdout log
#SBATCH --error=gr00t_finetune_%j.err         ## stderr log

module purge
#module load python-miniconda3/4.12.0         ## load Conda/Mamba
module load python-miniconda3
eval "$(conda shell.bash hook)"              ## enable `conda activate`
# conda activate gr00t                         ## your env name
source activate /projects/p32775/pythonenvs/gr00t

cd ~/Documents/Github/Isaac-GR00T            ## your project path

python scripts/gr00t_finetune.py \
   --dataset-path datasets/gr00t_so100_gold_pnp_60e_4c/ \
   --num-gpus 1 \
   --output-dir ~/so100-checkpoints \
   --max-steps 2000 \
   --data-config so100 \
   --video-backend torchvision_av \
   --report-to wandb