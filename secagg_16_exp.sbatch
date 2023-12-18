#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=192GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=secagg_16_exp
#SBATCH --output=secagg_16_exp.out
#SBATCH --mail-user=ml9027@nyu.edu
#SBATCH --mail-type=ALL

export HF_HOME="/scratch/ml9027/.cache"
export HF_DATASETS_CACHE="/scratch/ml9027/.cache"
export TRANSFORMERS_CACHE="/scratch/ml9027/.cache"
export NUM_CLIENTS=16

module purge
singularity exec \
  --nv --overlay /scratch/ml9027/my_env/overlay-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif  \
  /bin/bash -c "source /ext3/env.sh; conda activate /scratch/ml9027/lamp-main/penv; \
  export PYTHONNOUSERSITE=True; \
  cd /scratch/ml9027/ming/lamp-main; python3 flwr_server.py & \
  sleep 10; \
  for i in \$(seq \$NUM_CLIENTS); do
      python3 flwr_client.py &
    done
    wait"