#!/usr/bin/env python3
"""
Debug script to test what the model is returning for a specific image
"""

import asyncio
import base64
import ollama
import json

async def debug_model_response():
    """Test what the model returns for a specific image."""
    try:
        # Initialize Ollama client
        client = ollama.AsyncClient(host="http://localhost:11434")
        model = "qwen2.5vl:32b"
        
        # Load test image
        image_path = "/N/project/fads_ng/analyst_reports_visualizations/data/images_crop/key_99795608/page8.png"
        
        print(f"Loading image: {image_path}")
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        print(f"Image loaded, size: {len(image_base64)} bytes")
        
        # Use the improved prompt
        prompt = """Analyze this image to detect and categorize ALL charts, graphs, tables, and data visualizations.

For each data visualization found:
1. Extract the title/heading (main descriptive text above or near the visualization)
2. Classify the type:
   - "line_chart": Line graphs, trend lines, time series charts
   - "bar_chart": Bar charts, column charts, histograms
   - "pie_chart": Pie charts, donut charts, circular displays
   - "other": Tables, scatter plots, heat maps, or any other visualization
3. Provide confidence (0.0 to 1.0) in your classification
4. Brief description of what data is shown

Important instructions:
- Look for multiple visualizations in the same image
- Extract exact title text or use null if no clear title is visible
- Be precise with type classification based on the visual structure
- Focus on actual data visualizations, not decorative elements
- If you find a table, classify it as "other"

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

CRITICAL: Return ONLY the JSON, no other text, no explanations, no markdown formatting."""

        print("Sending request to Ollama...")
        response = await client.chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_base64]
                }
            ],
            options={
                "temperature": 0.1,
                "num_predict": 4096
            }
        )
        
        print("Response received!")
        if response and 'message' in response:
            response_content = response['message']['content']
            print("Raw response:")
            print("-" * 50)
            print(response_content)
            print("-" * 50)
            
            # Try to parse as JSON
            try:
                parsed = json.loads(response_content)
                print("✅ Successfully parsed as JSON!")
                print(f"Parsed content: {json.dumps(parsed, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse as JSON: {e}")
                
                # Try to find JSON in the response
                import re
                json_patterns = [
                    r'```json\s*(\{.*?\})\s*```',
                    r'```\s*(\{.*?\})\s*```',
                    r'\{.*?\}'
                ]
                
                for pattern in json_patterns:
                    matches = re.findall(pattern, response_content, re.DOTALL)
                    for match in matches:
                        try:
                            json_str = match if pattern == r'\{.*?\}' else match
                            parsed = json.loads(json_str)
                            print(f"✅ Found JSON in code block: {json.dumps(parsed, indent=2)}")
                            return
                        except json.JSONDecodeError:
                            continue
                
                print("No valid JSON found in response")
        else:
            print("No response content received")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_model_response())