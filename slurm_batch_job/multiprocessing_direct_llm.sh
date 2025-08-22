#!/bin/bash
#SBATCH -J direct_llm_multiprocessing
#SBATCH -A r01352
#SBATCH -o txt_logs/direct_llm_multiprocessing_%j.txt
#SBATCH -e err_logs/direct_llm_multiprocessing_%j.err
#SBATCH -p hopper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=vsatpute@iu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=120G
#SBATCH --gpus-per-node=1

# Move to the directory from where you submitted the job
module load python/gpu/3.11.5

cd /N/project/fads_ng/analyst_reports_visualizations/codes/direct_llm

echo 'chmod +x /N/project/fads_ng/ollama_setup/bin/ollama'
echo 'chmod +x /N/project/fads_ng/ollama_setup/lib/ollama'

echo 'export PATH=$PATH:/N/project/fads_ng/ollama_setup/bin' >> ~/.bashrc
source ~/.bashrc

pip install -r requirements.txt

# Source the deployment script with image mode for 4 H100 GPUs
source ollama_server_deployment_direct.sh image

# Wait a bit for all Ollama instances to start
sleep 30

# Run program/commands with multiprocessing config
python chart_detector_direct.py --config multiprocessing_config_direct.json