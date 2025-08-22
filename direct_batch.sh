#!/bin/bash
#SBATCH -J direct_llm_single_test
#SBATCH -A r01352
#SBATCH -o txt_logs/direct_llm_single_test_%j.txt
#SBATCH -e err_logs/direct_llm_single_test_%j.err
#SBATCH -p hopper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=vsatpute@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=120G
#SBATCH --gpus-per-node=1

# This batch job has been moved to the slurm_batch_job directory
# Please use slurm_batch_job/single_file_test_direct_llm.sh instead

echo "This batch job has been moved to the slurm_batch_job directory"
echo "Please use slurm_batch_job/single_file_test_direct_llm.sh instead"