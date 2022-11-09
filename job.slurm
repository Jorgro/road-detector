#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="road_detector"
#SBATCH --output=srun.out
#SBATCH --mail-user=joergen.rosager@gmail.com
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load CUDA/11.4.1
which python3
pip3 install --user torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirements.txt --user
pip3 install -U openmim
mim install mmcv-full
python3 ./tools/train.py /cluster/home/jorgro/road_detector/configs/omnimod/road_detector.py --work-dir ./runs
uname -a
