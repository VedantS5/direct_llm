#!/usr/bin/env python3
"""
Direct LLM V1 - Chart Detection using Ollama Vision Models
Direct implementation without ContextGem, using Ollama's vision capabilities.
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from PIL import Image as PILImage
import base64
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChartDetection:
    """Single chart/table detection result."""
    title: Optional[str]
    type: str  # line_chart, bar_chart, pie_chart, table, other
    confidence: float
    description: str

@dataclass
class ImageAnalysisResult:
    """Result of analyzing a single image."""
    image_path: str
    success: bool
    processing_time: float
    detections: List[ChartDetection]
    error: Optional[str] = None

@dataclass
class AnalysisSummary:
    """Summary of analysis results."""
    total_images: int
    successful_images: int
    total_detections: int
    chart_type_counts: Dict[str, int]
    average_processing_time: float

class DirectChartDetector:
    """Chart detector using direct Ollama vision capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.client = None
        
    async def initialize(self):
        """Initialize the Ollama client(s)."""
        try:
            # Get Ollama configuration
            ollama_config = self.config.get("ollama", {})
            model = ollama_config.get("model", "qwen2.5vl:32b")
            api_base = ollama_config.get("api_base", "http://localhost:11434")
            
            # Check if we're using multiprocessing mode
            self.multiprocessing_mode = self.config.get("multiprocessing", {}).get("enabled", False)
            
            if self.multiprocessing_mode:
                # Initialize multiple clients for different ports
                self.clients = []
                ports = self.config.get("multiprocessing", {}).get("ports", [11434, 11435, 11436, 11437])
                for port in ports:
                    client_api_base = f"http://localhost:{port}"
                    client = ollama.AsyncClient(host=client_api_base)
                    self.clients.append(client)
                    logger.info(f"Initialized Ollama client for {client_api_base} with model: {model}")
                self.model = model
            else:
                # Initialize single Ollama client
                self.client = ollama.AsyncClient(host=api_base)
                self.model = model
                logger.info(f"Initialized Ollama client with model: {model}")
                logger.info(f"Using API base: {api_base}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client(s): {e}")
            raise
    
    def _create_analysis_prompt(self) -> str:
        """Create the analysis prompt for charts and tables from config."""
        # Use configurable prompt if available, otherwise use default
        return self.config.get("prompt", """You are an expert data visualization analyst. Your task is to carefully examine images and identify ALL charts, graphs, tables, and data visualizations present.

Instructions:
1. Look for any data visualizations including charts, graphs, tables, plots, diagrams, etc.
2. For EACH visualization found, provide the following information:
   - Extract the exact title/heading text (look above, below, or near the visualization)
   - Classify the type as one of: "line_chart", "bar_chart", "pie_chart", "table", or "other"
   - Rate your confidence in the classification (0.0 to 1.0)
   - Provide a brief description of what data is shown

Types:
- "line_chart": Line graphs showing trends over time or relationships
- "bar_chart": Bar charts, column charts, histograms showing comparisons
- "pie_chart": Pie charts, donut charts showing proportions
- "table": Data tables, spreadsheets, or tabular data presentations
- "other": Scatter plots, heat maps, or any other visualization

CRITICAL INSTRUCTIONS:
- Find ALL visualizations, even small or partial ones
- If there's no clear title, use null for the title field
- Be thorough and precise in your analysis
- Return ONLY the JSON format specified below, nothing else

Return your response in this exact JSON format:
{
  "detections": [
    {
      "title": "exact title text or null",
      "type": "line_chart",
      "confidence": 0.95,
      "description": "brief description of what data is shown"
    }
  ]
}

Example of a good response:
{
  "detections": [
    {
      "title": "Quarterly Sales Performance 2023",
      "type": "line_chart",
      "confidence": 0.98,
      "description": "Line chart showing monthly sales figures across four quarters"
    },
    {
      "title": "Revenue by Product Category",
      "type": "table",
      "confidence": 0.95,
      "description": "Data table showing product categories and their revenue figures"
    }
  ]
}

Return ONLY the JSON, no other text, no explanations, no markdown code blocks.""")
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _extract_json_from_response(self, response_content: str) -> Optional[Dict]:
        """Extract JSON from model response, handling various formats."""
        if not response_content:
            return None
            
        # Clean the response content
        response_content = response_content.strip()
        
        # Try to parse as JSON directly
        try:
            parsed = json.loads(response_content)
            if isinstance(parsed, dict) and ('detections' in parsed or 'Detections' in parsed):
                # Normalize key case
                if 'Detections' in parsed:
                    parsed['detections'] = parsed.pop('Detections')
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in code blocks
        import re
        
        # Look for JSON in code blocks - more patterns
        json_patterns = [
            r'```(?:json)?\s*({.*?})\s*```',
            r'```\s*({.*?})\s*```',
            r'\{[^{]*"detections"[^{]*\{[^}]*\}[^}]*\}',
            r'\{[^{]*"Detections"[^{]*\{[^}]*\}[^}]*\}'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_content, re.DOTALL)
            for match in matches:
                try:
                    # If it's the plain JSON pattern, use the full match
                    json_str = match
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and ('detections' in parsed or 'Detections' in parsed):
                        # Normalize key case
                        if 'Detections' in parsed:
                            parsed['detections'] = parsed.pop('Detections')
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        # Try to find any valid JSON object in the response
        # Find all potential JSON objects
        start_positions = []
        for i in range(len(response_content)):
            if response_content[i] == '{':
                start_positions.append(i)
        
        # Try each starting position
        for start in reversed(start_positions):  # Try from the end first (more likely to be complete)
            # Try to parse JSON starting from this position
            for end in range(len(response_content), start, -1):
                try:
                    json_str = response_content[start:end]
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and ('detections' in parsed or 'Detections' in parsed):
                        # Normalize key case
                        if 'Detections' in parsed:
                            parsed['detections'] = parsed.pop('Detections')
                        return parsed
                except json.JSONDecodeError:
                    continue
            
        return None
    
    def _fallback_detection(self, response_content: str) -> List[ChartDetection]:
        """Fallback method to extract detections when JSON parsing fails."""
        detections = []
        
        # Clean the response
        response_content = response_content.strip()
        
        # If response is empty or very short, return empty
        if len(response_content) < 10:
            return detections
            
        # Try to parse as JSON even if it's not perfectly formatted
        try:
            # Look for JSON-like structures in the response
            import re
            
            # Pattern to find potential detections arrays
            detections_pattern = r'[\[\{][^\[\{]*"title"[^\]\}]*[^\[\{]*[\]\}]'
            matches = re.findall(detections_pattern, response_content, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                try:
                    # Try to parse each potential detection
                    potential_json = "{" + match + "}" if not match.startswith("{") else match
                    parsed = json.loads(potential_json)
                    
                    # Look for detections array or single detection
                    if isinstance(parsed, dict):
                        if "detections" in parsed:
                            for detection in parsed["detections"]:
                                if isinstance(detection, dict):
                                    title = detection.get("title")
                                    chart_type = detection.get("type", "other").lower()
                                    confidence = max(0.0, min(1.0, detection.get("confidence", 0.3)))
                                    description = detection.get("description", "")
                                    
                                    # Validate chart type
                                    valid_types = ["line_chart", "bar_chart", "pie_chart", "table", "other"]
                                    if chart_type not in valid_types:
                                        chart_type = "other"
                                    
                                    detections.append(ChartDetection(
                                        title=title,
                                        type=chart_type,
                                        confidence=confidence,
                                        description=description
                                    ))
                        elif "title" in parsed:
                            # Single detection
                            title = parsed.get("title")
                            chart_type = parsed.get("type", "other").lower()
                            confidence = max(0.0, min(1.0, parsed.get("confidence", 0.3)))
                            description = parsed.get("description", "")
                            
                            # Validate chart type
                            valid_types = ["line_chart", "bar_chart", "pie_chart", "table", "other"]
                            if chart_type not in valid_types:
                                chart_type = "other"
                            
                            detections.append(ChartDetection(
                                title=title,
                                type=chart_type,
                                confidence=confidence,
                                description=description
                            ))
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.debug(f"Fallback JSON parsing failed: {e}")
        
        # If still no detections, try regex-based extraction
        if not detections:
            try:
                # Look for patterns like "title: ... type: ... confidence: ..."
                import re
                
                # Pattern to match detection-like text
                pattern = r'(?:title|Title):\s*["\']?([^"\'\n\r]*)["\']?.*?(?:type|Type):\s*["\']?([^"\'\n\r]*)["\']?.*?(?:confidence|Confidence):\s*([0-9.]+)'
                matches = re.findall(pattern, response_content, re.DOTALL | re.IGNORECASE)
                
                for match in matches:
                    title, chart_type, confidence_str = match
                    title = title.strip() if title.strip() != "null" else None
                    chart_type = chart_type.lower().strip()
                    confidence = 0.3  # Default low confidence
                    
                    try:
                        confidence = float(confidence_str)
                        confidence = max(0.0, min(1.0, confidence))
                    except ValueError:
                        pass
                    
                    # Validate chart type
                    valid_types = ["line_chart", "bar_chart", "pie_chart", "table", "other"]
                    if chart_type not in valid_types:
                        # Try to infer type from text
                        if 'line' in chart_type or 'trend' in chart_type:
                            chart_type = "line_chart"
                        elif 'bar' in chart_type or 'column' in chart_type or 'histogram' in chart_type:
                            chart_type = "bar_chart"
                        elif 'pie' in chart_type or 'donut' in chart_type:
                            chart_type = "pie_chart"
                        elif 'table' in chart_type:
                            chart_type = "table"
                        else:
                            chart_type = "other"
                    
                    detections.append(ChartDetection(
                        title=title or None,
                        type=chart_type,
                        confidence=confidence,
                        description="Extracted via regex pattern matching"
                    ))
            except Exception as e:
                logger.debug(f"Regex fallback failed: {e}")
        
        # If still no detections, try very basic keyword matching
        if not detections:
            # Simple heuristic: look for phrases that might indicate charts/tables
            chart_indicators = [
                'chart', 'graph', 'table', 'figure', 'diagram', 
                'visualization', 'plot', 'histogram', 'pie', 'bar',
                'line', 'trend', 'scatter', 'heat map'
            ]
            
            # Look for title-like patterns
            lines = response_content.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 10 and len(line) < 200:  # Reasonable length
                    # Check if line contains chart indicators
                    if any(indicator in line.lower() for indicator in chart_indicators):
                        # Try to extract a title-like phrase
                        # Look for capitalized phrases that might be titles
                        title_candidates = re.findall(r'[A-Z][^.!?]*[.!?:]?$', line)
                        title = title_candidates[0].strip() if title_candidates else line[:50] + "..."
                        
                        # Remove trailing punctuation
                        title = re.sub(r'[.!?:]+$', '', title).strip()
                        
                        # Determine type based on keywords
                        chart_type = "other"  # default
                        if 'line' in line.lower() or 'trend' in line.lower():
                            chart_type = "line_chart"
                        elif 'bar' in line.lower() or 'column' in line.lower() or 'histogram' in line.lower():
                            chart_type = "bar_chart"
                        elif 'pie' in line.lower() or 'donut' in line.lower():
                            chart_type = "pie_chart"
                        elif 'table' in line.lower():
                            chart_type = "table"
                        
                        detections.append(ChartDetection(
                            title=title if title else None,
                            type=chart_type,
                            confidence=0.2,  # Very low confidence for this fallback
                            description=f"Fallback detection from response content: {line[:50]}..."
                        ))
        
        logger.debug(f"Fallback detection found {len(detections)} detections")
        return detections
    
    async def analyze_image(self, image_path: str) -> ImageAnalysisResult:
        """Analyze a single image for charts/tables using direct Ollama API."""
        start_time = datetime.now()
        
        try:
            # Load and prepare image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Convert image to base64 for Ollama
            image_base64 = self._image_to_base64(image_path)
            
            # Create prompt
            prompt = self._create_analysis_prompt()
            
            # Use Ollama chat API with vision capabilities
            logger.debug(f"Starting vision analysis for {image_path}")
            
            # Prepare the message with image
            message = {
                'role': 'user',
                'content': prompt,
                'images': [image_base64]
            }
            
            # Determine which client to use
            if hasattr(self, 'multiprocessing_mode') and self.multiprocessing_mode:
                # For multiprocessing, we'll determine the client based on the image path
                # This will be handled in the calling function
                client = self.client
            else:
                client = self.client
                
            logger.debug(f"Sending request to Ollama with model: {self.model}")
            response = await client.chat(
                model=self.model,
                messages=[message],
                options={
                    "temperature": self.config.get("ollama", {}).get("model_options", {}).get("temperature", 0.1),
                    "num_predict": self.config.get("ollama", {}).get("model_options", {}).get("max_tokens", 4096)
                }
            )
            
            logger.debug(f"Vision analysis completed for {image_path}")
            
            # Process results
            detections = []
            response_content = ""
            if response and 'message' in response:
                try:
                    # Extract content from response
                    response_content = response['message']['content']
                    response_length = len(response_content)
                    logger.debug(f"Raw response length: {response_length}")
                    logger.debug(f"Raw response preview: {response_content[:500]}...")
                    
                    # Check if response is empty or too short
                    if response_length < 10:
                        logger.warning(f"Response is very short ({response_length} chars) for {image_path}")
                    
                    # Try to extract JSON from the response
                    parsed_result = self._extract_json_from_response(response_content)
                    if parsed_result is None:
                        logger.warning(f"Could not extract valid JSON from response for {image_path}")
                        logger.debug(f"Full response: {response_content}")
                    else:
                        logger.debug(f"Successfully parsed JSON response for {image_path}")
                        raw_detections = parsed_result.get("detections", [])
                        logger.debug(f"Found {len(raw_detections)} detections in JSON response")
                        
                        for i, detection in enumerate(raw_detections):
                            try:
                                # Validate and normalize type
                                chart_type = detection.get("type", "other").lower()
                                valid_types = ["line_chart", "bar_chart", "pie_chart", "table", "other"]
                                if chart_type not in valid_types:
                                    logger.warning(f"Invalid chart type '{chart_type}' for detection {i} in {image_path}, using 'other'")
                                    chart_type = "other"
                                
                                # Clamp confidence to valid range
                                confidence = detection.get("confidence", 0.5)  # Default confidence
                                try:
                                    confidence = float(confidence)
                                    confidence = max(0.0, min(1.0, confidence))
                                except (ValueError, TypeError):
                                    logger.warning(f"Invalid confidence value '{confidence}' for detection {i} in {image_path}, using 0.5")
                                    confidence = 0.5
                                
                                # Extract title
                                title = detection.get("title")
                                if title == "null" or title == "":
                                    title = None
                                
                                detections.append(ChartDetection(
                                    title=title,
                                    type=chart_type,
                                    confidence=confidence,
                                    description=detection.get("description", "")
                                ))
                                logger.debug(f"Added detection {i}: {title} ({chart_type})")
                            except Exception as e:
                                logger.error(f"Error processing detection {i} for {image_path}: {e}")
                                continue
                        
                except Exception as e:
                    logger.error(f"Error processing result for {image_path}: {e}")
                    logger.error(f"Raw response: {response_content[:1000]}...")
            
            # If we still have no detections, try a fallback approach
            if not detections and response_content:
                logger.info(f"Trying fallback detection for {image_path}")
                fallback_detections = self._fallback_detection(response_content)
                if fallback_detections:
                    detections.extend(fallback_detections)
                    logger.info(f"Used fallback detection for {image_path}, found {len(fallback_detections)} detections")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Completed analysis for {image_path}: {len(detections)} detections found in {processing_time:.2f}s")
            
            # Log detection details
            if detections:
                for i, detection in enumerate(detections):
                    logger.debug(f"Detection {i}: {detection.title} ({detection.type}, {detection.confidence})")
            
            return ImageAnalysisResult(
                image_path=image_path,
                success=True,
                processing_time=processing_time,
                detections=detections
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Analysis failed for {image_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ImageAnalysisResult(
                image_path=image_path,
                success=False,
                processing_time=processing_time,
                detections=[],
                error=str(e)
            )
    
    async def analyze_image_with_client(self, image_path: str, client_index: int) -> ImageAnalysisResult:
        """Analyze a single image using a specific Ollama client."""
        # Set the client for this specific call
        self.client = self.clients[client_index]
        return await self.analyze_image(image_path)
    
    async def analyze_directory_multiprocessing(self, input_dir: Optional[str] = None, max_directories: Optional[int] = None) -> List[ImageAnalysisResult]:
        """Analyze images across multiple key directories using multiprocessing."""
        # Use configured input directory if not provided
        if input_dir is None:
            input_dir = self.config.get("directories", {}).get("input_images", "./input_images")
            logger.info(f"Using configured input directory: {input_dir}")
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Directory not found: {input_dir}")
        
        # Find key directories (those that match key_XXXX pattern)
        key_directories = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('key_')]
        
        # Sort directories to ensure consistent ordering
        key_directories.sort(key=lambda x: x.name)
        
        # Limit number of directories if specified
        if max_directories:
            key_directories = key_directories[:max_directories]
        
        if not key_directories:
            logger.warning(f"No key directories found in {input_dir}")
            return []
        
        logger.info(f"Found {len(key_directories)} key directories to process")
        
        # Distribute directories across the available clients
        results = []
        semaphore = asyncio.Semaphore(len(self.clients))  # Limit concurrent tasks
        
        async def process_directory_with_client(dir_path: Path, client_index: int):
            """Process a single directory with a specific client."""
            async with semaphore:
                logger.info(f"Processing directory {dir_path.name} with client {client_index}")
                
                # Find image files in this directory
                image_extensions = self.config.get("file_patterns", {}).get("image_extensions", ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'])
                image_files = []
                
                for ext in image_extensions:
                    image_files.extend(dir_path.glob(f"*{ext}"))
                    image_files.extend(dir_path.glob(f"*{ext.upper()}"))
                
                dir_results = []
                for image_file in image_files:
                    try:
                        result = await self.analyze_image_with_client(str(image_file), client_index)
                        dir_results.append(result)
                        logger.info(f"Processed {image_file.name} from {dir_path.name}: {len(result.detections)} detections")
                    except Exception as e:
                        logger.error(f"Failed to process {image_file} from {dir_path.name}: {e}")
                        dir_results.append(ImageAnalysisResult(
                            image_path=str(image_file),
                            success=False,
                            processing_time=0.0,
                            detections=[],
                            error=str(e)
                        ))
                
                return dir_results
        
        # Create tasks for all directories
        tasks = []
        for i, directory in enumerate(key_directories):
            client_index = i % len(self.clients)  # Distribute evenly across clients
            task = process_directory_with_client(directory, client_index)
            tasks.append(task)
        
        # Process all directories concurrently
        directory_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        for dir_result in directory_results:
            if isinstance(dir_result, Exception):
                logger.error(f"Directory processing failed: {dir_result}")
                continue
            results.extend(dir_result)
        
        return results
    
    async def analyze_directory(self, input_dir: Optional[str] = None, max_images: Optional[int] = None) -> List[ImageAnalysisResult]:
        """Analyze all images in a directory."""
        # Use configured input directory if not provided
        if input_dir is None:
            input_dir = self.config.get("directories", {}).get("input_images", "./input_images")
            logger.info(f"Using configured input directory: {input_dir}")
        
        # Handle target key for specific directory processing
        file_patterns = self.config.get("file_patterns", {})
        target_key = file_patterns.get("target_key")
        if target_key and not input_dir.endswith(target_key):
            input_dir = os.path.join(input_dir, target_key)
            logger.info(f"Targeting specific key directory: {input_dir}")
        
        # Use configured max_images if not provided
        if max_images is None:
            config_max = self.config.get("processing_settings", {}).get("max_images")
            if config_max is not None:
                max_images = config_max
                logger.info(f"Using configured max_images: {max_images}")
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Directory not found: {input_dir}")
        
        # Find image files
        image_extensions = self.config.get("file_patterns", {}).get("image_extensions", ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'])
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"**/*{ext}"))
            image_files.extend(input_path.glob(f"**/*{ext.upper()}"))
        
        if max_images:
            image_files = image_files[:max_images]
        
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return []
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images sequentially for server deployment
        results = []
        for image_file in image_files:
            try:
                result = await self.analyze_image(str(image_file))
                results.append(result)
                logger.info(f"Processed {image_file.name}: {len(result.detections)} detections")
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
                results.append(ImageAnalysisResult(
                    image_path=str(image_file),
                    success=False,
                    processing_time=0.0,
                    detections=[],
                    error=str(e)
                ))
        
        return results
    
    def generate_summary(self, results: List[ImageAnalysisResult]) -> AnalysisSummary:
        """Generate summary statistics from results."""
        total_images = len(results)
        successful_images = sum(1 for r in results if r.success)
        
        # Count detections by type
        type_counts = {"line_chart": 0, "bar_chart": 0, "pie_chart": 0, "table": 0, "other": 0}
        total_detections = 0
        
        for result in results:
            for detection in result.detections:
                type_counts[detection.type] += 1
                total_detections += 1
        
        # Calculate average processing time
        processing_times = [r.processing_time for r in results if r.processing_time > 0]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        return AnalysisSummary(
            total_images=total_images,
            successful_images=successful_images,
            total_detections=total_detections,
            chart_type_counts=type_counts,
            average_processing_time=avg_processing_time
        )
    
    def save_results(self, results: List[ImageAnalysisResult], output_path: str = None) -> str:
        """Save results to JSON file."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"chart_detection_results_{timestamp}.json"
        
        # Use configured output directory if available
        output_dir = self.config.get("directories", {}).get("output", "./output")
        if not os.path.isabs(output_path):
            output_path = os.path.join(output_dir, output_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        summary = self.generate_summary(results)
        
        output_data = {
            "summary": asdict(summary),
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "results": []
        }
        
        for result in results:
            result_data = {
                "image_path": result.image_path,
                "success": result.success,
                "processing_time": result.processing_time,
                "detections": [asdict(d) for d in result.detections]
            }
            
            if not result.success:
                result_data["error"] = result.error
            
            output_data["results"].append(result_data)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path

    def save_results_csv(self, results: List[ImageAnalysisResult], output_path: str = None) -> str:
        """Save results to CSV file."""
        import csv
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"chart_detection_results_{timestamp}.csv"
        
        # Use configured output directory if available
        output_dir = self.config.get("directories", {}).get("output", "./output")
        if not os.path.isabs(output_path):
            output_path = os.path.join(output_dir, output_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Define CSV headers with new filename and page_number columns
        headers = ["filename", "page_number", "image_path", "success", "processing_time", "detection_title", "detection_type", "confidence", "description", "error"]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for result in results:
                # Extract filename and page number from image path
                image_path = result.image_path
                # Get the basename of the file (e.g., page8.png)
                filename_base = os.path.basename(image_path)
                # Extract the filename part without extension (e.g., page8)
                filename_without_ext = os.path.splitext(filename_base)[0]
                # Extract page number from filename (assuming format is "pageX")
                page_number = ""
                if filename_without_ext.startswith("page") and filename_without_ext[4:].isdigit():
                    page_number = filename_without_ext[4:]
                
                # Extract the directory name which contains the key (e.g., key_99795608)
                dir_name = os.path.basename(os.path.dirname(image_path))
                
                # Only write rows for images that have detections
                if result.detections:
                    for detection in result.detections:
                        writer.writerow([
                            dir_name,  # filename (key directory name)
                            page_number,  # page_number
                            result.image_path,
                            result.success,
                            result.processing_time,
                            detection.title or "",
                            detection.type,
                            detection.confidence,
                            detection.description,
                            result.error or ""
                        ])
                # If there are no detections, we simply don't write any rows for this image
        
        logger.info(f"Results saved to: {output_path}")
        return output_path

# Server CLI
async def main():
    """Main CLI function for server deployment."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Direct LLM Chart Detector - Ollama Vision')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--input', help='Input directory (overrides config)')
    parser.add_argument('--max-images', type=int, help='Maximum images to process (overrides config)')
    parser.add_argument('--output', help='Output file path (CSV or JSON)')
    parser.add_argument('--output-format', choices=['json', 'csv'], default='csv', help='Output format')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded configuration from: {args.config}")
    except Exception as e:
        print(f"❌ Failed to load config from {args.config}: {e}")
        sys.exit(1)
    
    # Enable verbose logging if requested
    if args.verbose or config.get("processing_settings", {}).get("verbose", False):
        logging.getLogger().setLevel(logging.DEBUG)
        print("🔍 Verbose logging enabled")
    
    print(f"🚀 Starting Direct LLM Chart Detection (Ollama Vision)")
    print(f"📁 Config: {args.config}")
    
    # Initialize detector
    detector = DirectChartDetector(config)
    await detector.initialize()
    
    # Determine input directory
    input_dir = args.input
    if input_dir:
        print(f"📁 Input directory (CLI): {input_dir}")
    else:
        config_input = config.get("directories", {}).get("input_images")
        print(f"📁 Input directory (config): {config_input}")
    
    # Determine max images
    max_images = args.max_images
    if max_images:
        print(f"📊 Max images (CLI): {max_images}")
    else:
        config_max = config.get("processing_settings", {}).get("max_images")
        if config_max:
            print(f"📊 Max images (config): {config_max}")
        else:
            print(f"📊 Max images: unlimited")
    
    # Process images
    try:
        print(f"\n⚡ Starting image analysis...")
        
        # Check if multiprocessing mode is enabled
        multiprocessing_enabled = config.get("multiprocessing", {}).get("enabled", False)
        max_directories = config.get("multiprocessing", {}).get("max_directories")
        
        if multiprocessing_enabled:
            print("🔄 Using multiprocessing mode")
            results = await detector.analyze_directory_multiprocessing(input_dir, max_directories)
        else:
            results = await detector.analyze_directory(input_dir, max_images)
        
        if not results:
            print("⚠️  No images processed")
            return
        
        # Generate summary
        summary = detector.generate_summary(results)
        
        # Save results
        if args.output_format == 'csv':
            output_file = detector.save_results_csv(results, args.output)
        else:
            output_file = detector.save_results(results, args.output)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Print summary
        print(f"\n📊 Analysis Summary:")
        print(f"   Images processed: {summary.total_images}")
        print(f"   Successful: {summary.successful_images}")
        print(f"   Total detections: {summary.total_detections}")
        print(f"   Average time: {summary.average_processing_time:.2f}s")
        
        for chart_type, count in summary.chart_type_counts.items():
            if count > 0:
                print(f"   {chart_type}: {count}")
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())