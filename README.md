# Direct LLM - Chart Detection using Ollama Vision

Direct implementation of chart and table detection from images using Ollama's vision capabilities, without ContextGem dependency.

## 🚀 Quick Start

There are two ways to run this code:

### 1. Interactive Mode

First, get a machine with the required resources:
```bash
srun -p hopper --cpus-per-task 20 --gpus-per-node 1 --mem 40GB -A r01352 --time 1:00:00 --pty bash
```

Then load the required modules:
```bash
module load python/gpu/3.11.5
```

Source the Ollama server deployment script:
```bash
source ollama_server_deployment_direct.sh image
```

Finally, run the chart detector:
```bash
python3 chart_detector_direct.py --config multiprocessing_config_direct.json
```

### 2. Batch Job Mode

Navigate to the slurm_batch_job directory and submit the batch job:
```bash
cd slurm_batch_job
sbatch multiprocessing_direct_llm.sh
```

This will run on 4 H100 GPUs with exactly the same configuration as described in interactive mode.

## ⚙️ Configuration Parameters

The configuration files control all aspects of the analysis. Here are the available parameters and options:

### Directories
- `directories.input_images`: Path to input images directory
- `directories.output`: Path to output directory

### Ollama Settings
- `ollama.model`: Vision model to use (default: qwen2.5vl:32b)
- `ollama.api_base`: Ollama API base URL
- `ollama.model_options.temperature`: Model temperature (default: 0.1)
- `ollama.model_options.max_tokens`: Maximum tokens for response (default: 4096)

### File Patterns
- `file_patterns.image_extensions`: List of image file extensions to process
- `file_patterns.target_key`: Specific subdirectory to process (optional)

### Processing Settings
- `processing_settings.max_images`: Maximum number of images to process
- `processing_settings.verbose`: Enable verbose logging

### Analysis Prompt
- `prompt`: Custom analysis prompt for the vision model

### Multiprocessing Options
- `multiprocessing.enabled`: Enable multiprocessing mode
- `multiprocessing.ports`: List of ports for Ollama instances
- `multiprocessing.max_directories`: Maximum number of directories to process

## 📁 Project Structure

```
direct_llm/
├── chart_detector_direct.py       # Main server implementation (ONE CODE FILE)
├── main_config_direct.json       # Main configuration for all files (copied from main project)
├── single_test_config_direct.json # Test configuration for specific files (copied from main project)
├── multiprocessing_config_direct.json # Multiprocessing configuration for 4 H100 GPUs
├── requirements.txt              # Python dependencies (no ContextGem)
├── ollama_server_deployment_direct.sh # Server deployment script
├── slurm_batch_job/              # SLURM batch job scripts and logs
│   ├── single_file_test_direct_llm.sh # Single file test batch job
│   ├── multiprocessing_direct_llm.sh # Multiprocessing batch job for 4 GPUs
│   ├── txt_logs/                 # Output logs directory
│   └── err_logs/                 # Error logs directory
└── README.md                     # This file
```

## 🎯 Usage Examples

### Process with different configurations:
```bash
# Test specific key files (up to 11 images)
python3 chart_detector_direct.py --config single_test_config_direct.json

# Process all files
python3 chart_detector_direct.py --config main_config_direct.json

# Process files using multiprocessing across 4 H100 GPUs
python3 chart_detector_direct.py --config multiprocessing_config_direct.json

# Override max images from command line
python3 chart_detector_direct.py --config main_config_direct.json --max-images 5

# Custom input directory
python3 chart_detector_direct.py --config main_config_direct.json --input /path/to/images

# Verbose output
python3 chart_detector_direct.py --config main_config_direct.json --verbose
```

## 📊 Output

Results are saved as JSON or CSV with:
- Detection summary (counts, processing time)
- Individual image results
- Chart titles, types, and confidence scores

### CSV Format

The CSV output includes the following columns:
- `filename`: The name of the file (directory) being processed (e.g., key_99795608)
- `page_number`: The page number extracted from the image filename (e.g., 8 for page8.png)
- `image_path`: Full path to the image file
- `success`: Whether the analysis was successful
- `processing_time`: Time taken to process the image
- `detection_title`: Title of the detected chart/table
- `detection_type`: Type of visualization (line_chart, bar_chart, pie_chart, table, other)
- `confidence`: Confidence score (0.0 to 1.0)
- `description`: Description of the detected visualization
- `error`: Any error messages (if applicable)

## 🖥️ SLURM Batch Jobs

The project includes pre-configured SLURM batch jobs for both single file testing and multiprocessing across 4 GPUs:

### Single File Test Job
```bash
# Submit the single file test job
sbatch slurm_batch_job/single_file_test_direct_llm.sh
```

### Multiprocessing Job (4 GPUs)
```bash
# Submit the multiprocessing job for 4 H100 GPUs
sbatch slurm_batch_job/multiprocessing_direct_llm.sh
```

Output and error logs are stored in `slurm_batch_job/txt_logs/` and `slurm_batch_job/err_logs/` respectively, with job IDs in the filenames.

## 🔧 Key Differences from ContextGem Version

1. **No ContextGem dependency**: Uses direct Ollama API calls
2. **Configurable prompt**: Prompt can be customized in the configuration files
3. **Simplified dependencies**: Only requires `ollama` and `pillow` packages
4. **Same input/output format**: Maintains compatibility with existing workflows