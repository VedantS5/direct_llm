# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Direct LLM is a chart and table detection system that uses Ollama vision models to analyze images. It's designed for server deployment with configurable GPU setups and processes large volumes of images for financial analyst report visualization analysis.

**Key Characteristics:**
- Single-file Python implementation (`chart_detector_direct.py`)
- Server-first architecture (never test locally)
- Configuration-driven approach with JSON configs
- Direct Ollama API integration (no ContextGem dependency)
- Supports multiple GPU configurations and parallel processing

## Common Commands

### Server Setup and Deployment
```bash
# Run Ollama server in single mode (for development/testing)
./ollama_server_deployment_direct.sh single

# Auto-detect GPU type and configure accordingly
./ollama_server_deployment_direct.sh

# Manual GPU type specification
./ollama_server_deployment_direct.sh h100  # or v100, a100, qwq, image
```

### SLURM Batch Jobs
```bash
# Submit the single file test job
sbatch slurm_batch_job/single_file_test_direct_llm.sh

# Submit the multiprocessing job for 4 H100 GPUs
sbatch slurm_batch_job/multiprocessing_direct_llm.sh
```

### Chart Detection Analysis
```bash
# Test on specific key directory with limited images
python3 chart_detector_direct.py --config single_test_config_direct.json

# Process all images with main configuration
python3 chart_detector_direct.py --config main_config_direct.json

# Process images using multiprocessing across 4 H100 GPUs
python3 chart_detector_direct.py --config multiprocessing_config_direct.json

# Override configuration options
python3 chart_detector_direct.py --config main_config_direct.json --max-images 5
python3 chart_detector_direct.py --config main_config_direct.json --input /path/to/images
python3 chart_detector_direct.py --config main_config_direct.json --verbose
python3 chart_detector_direct.py --config main_config_direct.json --output-format csv
```

### Dependencies and Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Essential packages only
pip install ollama pillow
```

## Architecture and Code Structure

### Single-File Design Philosophy
The entire Python implementation is contained in `chart_detector_direct.py`. This deliberate architectural choice:
- Simplifies server deployment
- Reduces dependency complexity
- Follows the project's "single file" principle from QWEN.md guidelines

### Core Classes
- **`DirectChartDetector`**: Main orchestrator class handling Ollama client initialization, image analysis, and result processing
- **`ChartDetection`**: Data class for individual chart/table detection results
- **`ImageAnalysisResult`**: Data class for per-image analysis outcomes
- **`AnalysisSummary`**: Data class for aggregated statistics across all processed images

### Configuration System
The project uses JSON configuration files instead of code modifications:
- `main_config_direct.json`: Production configuration for processing all images
- `single_test_config_direct.json`: Test configuration for limited image processing
- `multiprocessing_config_direct.json`: Multiprocessing configuration for parallel processing across 4 H100 GPUs
- Configurable prompts, model settings, directories, and processing parameters

### GPU and Server Architecture
- **Multi-GPU Support**: The deployment script auto-detects and configures for V100, A100, H100, and other GPU types
- **Parallel Ollama Instances**: Multiple Ollama server instances run on different ports across GPUs
- **Server-First Design**: All development and testing assumes server/remote machine deployment

### Image Processing Pipeline
1. **Initialization**: Ollama client setup with configured model (`qwen2.5vl:32b` by default)
2. **Image Analysis**: Base64 encoding → Vision model processing → JSON response parsing
3. **Fallback Handling**: Multiple parsing strategies for robust response extraction
4. **Result Aggregation**: Statistics generation and output in CSV/JSON formats

## Development Guidelines

### Core Principles

#### 1. Server-First Development
- **Never test on your local machine**
- **Assume the code will always run on a server or remote machine**
- This ensures consistency and avoids environment-specific issues
- All development and testing assumes server/remote machine deployment

#### 2. Code Structure
- **Keep actual Python code in a single file as much as possible**
- This simplifies deployment and reduces complexity
- The entire implementation is contained in `chart_detector_direct.py`
- Single-file design follows the project's core architectural principle

#### 3. Configuration Management
- **Keep all options in configuration files**
- **Create new configuration versions instead of modifying code for small changes**
- This allows for easy experimentation and rollbacks
- Modify JSON configuration files for different processing scenarios
- Use `target_key` in file patterns to process specific subdirectories
- Adjust `max_images` for testing vs. production runs
- Configure custom prompts for different analysis requirements

#### 4. Documentation
- **Keep the README up-to-date with the latest changes**
- **Document any new features, configurations, or usage instructions**
- Maintain clear documentation for server deployment procedures

#### 5. Output Format
- **Default output format should be CSV**
- **All JSON configuration files should specify CSV as the default output format**
- Results saved as JSON (detailed) or CSV (tabular) formats with automatic timestamping

### Working with Configurations
- Modify JSON configuration files for different processing scenarios
- Use `target_key` in file patterns to process specific subdirectories
- Adjust `max_images` for testing vs. production runs
- Configure custom prompts for different analysis requirements

### Error Handling and Debugging
- The system includes comprehensive fallback detection mechanisms
- Multiple JSON parsing strategies handle various model response formats
- Verbose logging available via `--verbose` flag or config setting
- Processing continues even if individual images fail

### Output and Results
- Results saved as JSON (detailed) or CSV (tabular) formats
- Automatic timestamped output files
- Summary statistics include processing times and detection type counts
- Individual image results preserved with error details
- CSV output includes additional `filename` and `page_number` columns for better context

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
