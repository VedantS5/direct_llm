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

# Move to the directory from where you submitted the job
module load python/gpu/3.11.5

cd /N/project/fads_ng/analyst_reports_visualizations/codes/direct_llm

echo 'chmod +x /N/project/fads_ng/ollama_setup/bin/ollama'
echo 'chmod +x /N/project/fads_ng/ollama_setup/lib/ollama'

echo 'export PATH=$PATH:/N/project/fads_ng/ollama_setup/bin' >> ~/.bashrc
source ~/.bashrc

pip install -r requirements.txt

source ollama_server_deployment_direct.sh single

# Run program/commands
python chart_detector_direct.py --config single_test_config_direct.json