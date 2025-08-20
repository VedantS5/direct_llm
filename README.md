# Direct LLM - Chart Detection using Ollama Vision

Direct implementation of chart and table detection from images using Ollama's vision capabilities, without ContextGem dependency.

## 🚀 Quick Start

### 1. Server Setup
```bash
# Run Ollama server in single mode
./ollama_server_deployment_direct.sh single
```

This script will automatically:
- Install Python dependencies from `requirements.txt`
- Pull the required vision model (`qwen2.5vl:32b`)

### 2. Test on Specific Files
```bash
# Test on specific key directory with up to 11 images
python3 chart_detector_direct.py --config single_test_config_direct.json
```

### 3. Process All Images
```bash
# Process all images with main configuration
python3 chart_detector_direct.py --config main_config_direct.json
```

## 📁 Project Structure

```
direct_llm/
├── chart_detector_direct.py       # Main server implementation (ONE CODE FILE)
├── main_config_direct.json       # Main configuration for all files (copied from main project)
├── single_test_config_direct.json # Test configuration for specific files (copied from main project)
├── requirements.txt              # Python dependencies (no ContextGem)
├── ollama_server_deployment_direct.sh # Server deployment script
└── README.md                     # This file
```

## ⚙️ Configurations

Configuration files are direct copies from the main project with these changes:
1. `contextgem` section replaced with `ollama` section
2. Added configurable `prompt` field for analysis instructions

## 🎯 Usage Examples

### Process with different configurations:
```bash
# Test specific key files (up to 11 images)
python3 chart_detector_direct.py --config single_test_config_direct.json

# Process all files
python3 chart_detector_direct.py --config main_config_direct.json

# Override max images from command line
python3 chart_detector_direct.py --config main_config_direct.json --max-images 5

# Custom input directory
python3 chart_detector_direct.py --config main_config_direct.json --input /path/to/images

# Verbose output
python3 chart_detector_direct.py --config main_config_direct.json --verbose
```

## 📊 Output

Results are saved as JSON with:
- Detection summary (counts, processing time)
- Individual image results
- Chart titles, types, and confidence scores

## 🔧 Key Differences from ContextGem Version

1. **No ContextGem dependency**: Uses direct Ollama API calls
2. **Configurable prompt**: Prompt can be customized in the configuration files
3. **Simplified dependencies**: Only requires `ollama` and `pillow` packages
4. **Same input/output format**: Maintains compatibility with existing workflows