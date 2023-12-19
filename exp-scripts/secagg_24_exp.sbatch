#!/bin/bash
#SBATCH --array=0-24
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=secagg_24_exp_2
#SBATCH --output=secagg_24_exp_2_%a.out
#SBATCH --mail-user=ml9027@nyu.edu
#SBATCH --mail-type=ALL

export HF_HOME="/scratch/ml9027/.cache"
export HF_DATASETS_CACHE="/scratch/ml9027/.cache"
export TRANSFORMERS_CACHE="/scratch/ml9027/.cache"

export MASTER_PORT=10722
export MASTER_ADDR_FILE="/scratch/ml9027/ming/lamp-main/master_addr-24.txt"

module purge

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    # Run the server on the first task
    echo "$(hostname -s).hpc.nyu.edu" > $MASTER_ADDR_FILE
    singularity exec \
      --overlay /scratch/ml9027/my_env/overlay-15GB-500K.ext3:ro \
                /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif  \
        /bin/bash -c "source /ext3/env.sh; conda activate /scratch/ml9027/lamp-main/penv; \
      export PYTHONNOUSERSITE=True; \
      cd /scratch/ml9027/ming/lamp-main; python3 flwr_server.py"
else
    # Run the clients on the other tasks
    while [ ! -f $MASTER_ADDR_FILE ]; do sleep 1; done
    export MASTER_ADDR=$(cat $MASTER_ADDR_FILE)
    singularity exec \
      --overlay /scratch/ml9027/my_env/overlay-15GB-500K.ext3:ro \
                /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif  \
        /bin/bash -c "source /ext3/env.sh; conda activate /scratch/ml9027/lamp-main/penv; \
      export PYTHONNOUSERSITE=True; \
      cd /scratch/ml9027/ming/lamp-main; python3 flwr_client.py"
fi