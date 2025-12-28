"""Streamlit app for SAM3 API visualization."""
import argparse
import base64
import io
import json
import logging
import sys
import time
from typing import Optional

import requests
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

# Configure streamlit page
st.set_page_config(layout="wide", page_title="SAM3 Segmentation Demo")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--api-url", default="http://localhost:8000", help="SAM3 API URL")
args, _ = parser.parse_known_args()
DEFAULT_API_URL = args.api_url


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def decode_base64_to_image(b64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    img_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_data))


def visualize_masks(original_image: Image.Image, masks_b64: list[str]) -> Image.Image:
    """Overlay masks on original image with light green fill and yellow border."""
    # Convert to RGBA for transparency
    result = original_image.convert("RGBA")
    
    for i, mask_b64 in enumerate(masks_b64):
        # Decode mask
        mask_img = decode_base64_to_image(mask_b64)
        mask_array = np.array(mask_img) > 0
        
        # Create overlay with light green fill (144, 238, 144 is light green)
        overlay = np.zeros((*mask_array.shape[:2], 4), dtype=np.uint8)
        overlay[mask_array] = [144, 238, 144, 100]  # Light green with transparency
        
        # Add yellow border (dilate then subtract to get border)
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(mask_array, iterations=2)
        border = dilated & ~mask_array
        overlay[border] = [255, 255, 0, 200]  # Yellow border, more opaque
        
        # Blend with result
        overlay_img = Image.fromarray(overlay, mode="RGBA")
        result = Image.alpha_composite(result, overlay_img)
    
    return result


def call_sam3_api(image: Image.Image, prompt: Optional[str] = None, boxes: Optional[list] = None, api_url: str = None) -> dict:
    """Call SAM3 API endpoint."""
    # Encode image
    image_b64 = encode_image_to_base64(image)
    
    # Build request
    payload = {
        "image": image_b64,
        "response_format": "b64_json"
    }
    
    if prompt:
        payload["prompt"] = prompt
    
    if boxes:
        payload["boxes"] = boxes
    
    # Log request
    logger.info(f"Sending request to {api_url}/sam3")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Boxes: {boxes}")
    logger.info(f"Image size: {image.size}")
    
    # Debug: log payload structure (without full image base64)
    debug_payload = {k: v if k != "image" else f"<base64 image {len(v)} chars>" for k, v in payload.items()}
    logger.debug(f"Full request payload structure: {json.dumps(debug_payload, indent=2)}")
    
    # Call API
    try:
        start_time = time.time()
        response = requests.post(f"{api_url}/sam3", json=payload, timeout=60)
        elapsed_time = time.time() - start_time
        
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response time: {elapsed_time:.2f} seconds")
        
        # Log raw response text (truncated if too long)
        raw_response = response.text
        logger.debug(f"Raw response (first 1000 chars): {raw_response[:1000]}")
        
        response.raise_for_status()
        
        result = response.json()
        num_masks = len(result.get('data', []))
        logger.info(f"✓ Successfully received {num_masks} mask(s) in {elapsed_time:.2f}s")
        
        # Debug: log response structure (without full mask base64)
        debug_result = {
            "created": result.get("created"),
            "data": [
                {k: v if k != "b64_json" else f"<base64 mask {len(v)} chars>" for k, v in item.items()}
                for item in result.get("data", [])
            ]
        }
        logger.debug(f"Full response structure: {json.dumps(debug_result, indent=2)}")
        
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        raise


st.title("SAM3 Segmentation Demo")

st.markdown("""
Upload an image and provide a text prompt or bounding box to segment objects using SAM3.
""")

# API URL input
API_URL = st.text_input("API URL", value=DEFAULT_API_URL, help="URL of the SAM3 API server")

# Image upload
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, width='stretch')
    
    # Prompt input
    st.subheader("Segmentation Options")
    
    prompt_type = st.radio("Prompt Type", ["Text", "Bounding Box"])
    
    text_prompt = None
    boxes = None
    
    if prompt_type == "Text":
        text_prompt = st.text_input("Text Prompt", placeholder="e.g., 'a person', 'the dog'")
    else:
        st.markdown("Enter bounding box coordinates (normalized 0-1):")
        box_col1, box_col2, box_col3, box_col4 = st.columns(4)
        with box_col1:
            x1 = st.number_input("x1", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        with box_col2:
            y1 = st.number_input("y1", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        with box_col3:
            x2 = st.number_input("x2", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
        with box_col4:
            y2 = st.number_input("y2", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
        
        # Convert to center-width-height format expected by API
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        boxes = [{"cx": cx, "cy": cy, "w": w, "h": h, "label": True}]
    
    # Segment button
    if st.button("Segment", type="primary"):
        if not text_prompt and not boxes:
            st.error("Please provide either a text prompt or bounding box")
        else:
            try:
                with st.spinner("Segmenting..."):
                    logger.info("=== Starting segmentation ===")
                    result = call_sam3_api(image, prompt=text_prompt, boxes=boxes, api_url=API_URL)
                    logger.info("=== Segmentation complete ===")
                
                # Extract masks
                masks = [item["b64_json"] for item in result["data"]]
                
                if masks:
                    # Visualize
                    with col2:
                        st.subheader("Segmentation Result")
                        visualized = visualize_masks(image, masks)
                        st.image(visualized, use_container_width=True)
                    
                    st.success(f"Found {len(masks)} mask(s)")
                    
                    # Show details
                    with st.expander("View Details"):
                        for i, item in enumerate(result["data"]):
                            st.json({
                                "mask_id": i,
                                "score": item.get("score"),
                                "bbox": item.get("bbox"),
                                "revised_prompt": item.get("revised_prompt")
                            })
                else:
                    st.warning("No masks found")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"API Error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")

