"""FastAPI application for serving SAM3 model."""

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image

from app.models import SAM3Request, SAM3Response, ImageData
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Override existing configuration
)
logger = logging.getLogger(__name__)
# Also set uvicorn's loggers to show our logs
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# Global model instance
model_state: Dict = {}


def get_bpe_path() -> str:
    """Get the path to the BPE vocabulary file."""
    import sam3
    sam3_root = os.path.dirname(sam3.__file__)
    return f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"


def load_model():
    """Load SAM3 model and processor.
    
    Requires HF_TOKEN environment variable for accessing the gated SAM3 model.
    """
    import os
    
    # Check for HuggingFace token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable not found. "
            "SAM3 is a gated model that requires authentication. "
            "Get a token at https://huggingface.co/settings/tokens "
            "and request access to facebook/sam3"
        )
    
    # Login to HuggingFace
    from huggingface_hub import login
    try:
        login(token=hf_token, add_to_git_credential=False)
        logger.info("Successfully authenticated with HuggingFace")
    except Exception as e:
        raise RuntimeError(f"HuggingFace authentication failed: {e}")
    
    # Enable TF32 for Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    bpe_path = get_bpe_path()
    model = build_sam3_image_model(bpe_path=bpe_path)
    
    # Enable autocast for bfloat16
    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model loading."""
    # Startup: load model
    logger.info("Loading SAM3 model...")
    model_state["model"] = load_model()
    logger.info("SAM3 model loaded successfully!")
    yield
    # Shutdown: cleanup
    model_state.clear()


# Create FastAPI app
app = FastAPI(
    title="SAM3 API",
    description="API for SAM 3 (Segment Anything Model 3) - text and visual prompting for image segmentation",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with sanitized logging."""
    body = await request.body()
    try:
        body_json = body.decode('utf-8')
        import json
        body_dict = json.loads(body_json)
        # Truncate base64 fields for logging
        sanitized = {k: f"<base64 {len(v)} chars>" if k == "image" and len(str(v)) > 100 else v 
                    for k, v in body_dict.items()}
        logger.error(f"Validation error on {request.method} {request.url.path}")
        logger.error(f"Validation errors: {exc.errors()}")
        logger.error(f"Request payload: {json.dumps(sanitized, indent=2)}")
    except:
        logger.error(f"Validation error on {request.method} {request.url.path}")
        logger.error(f"Validation errors: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        # Remove data URI prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """Encode binary mask to base64 PNG string."""
    # Convert boolean mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Encode as PNG
    success, buffer = cv2.imencode(".png", mask_uint8)
    if not success:
        raise ValueError("Failed to encode mask as PNG")
    
    # Convert to base64
    base64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return base64_str


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SAM3 API",
        "version": "0.1.0",
        "endpoints": {
            "/sam3": "POST - Segment objects in images using text or visual prompts",
            "/health": "GET - Health check endpoint"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    model_loaded = "model" in model_state and model_state["model"] is not None
    return {
        "status": "healthy" if model_loaded else "model not loaded",
        "model_loaded": model_loaded,
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/sam3", response_model=SAM3Response)
async def segment_image(request: SAM3Request):
    """
    Segment objects in an image using SAM3.
    
    OpenAI-compatible API for image segmentation with text or visual prompts.
    Similar to OpenAI's image edit endpoint, but specialized for segmentation tasks.
    
    Supports:
    - Text prompts: Describe what to segment (e.g., "person", "face", "shoe")
    - Visual prompts: Provide bounding boxes as examples
    - Multiple results: Use 'n' parameter to get multiple mask variations
    """
    start_time = time.time()
    
    logger.info("=== Received POST /sam3 request ===")
    logger.info(f"Request has prompt: {bool(request.prompt)}")
    logger.info(f"Request has boxes: {bool(request.boxes)}")
    if request.prompt:
        logger.info(f"Text prompt: {request.prompt}")
    if request.boxes:
        logger.info(f"Number of boxes: {len(request.boxes)}")
    
    if not request.prompt and not request.boxes:
        logger.error("No prompt or boxes provided")
        raise HTTPException(
            status_code=400,
            detail="At least one of 'prompt' (text) or 'boxes' (visual prompts) must be provided"
        )
    
    model = model_state.get("model")
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        decode_start = time.time()
        image = decode_base64_image(request.image)
        width, height = image.size
        logger.info(f"Decoded image: {width}x{height} pixels (took {time.time() - decode_start:.2f}s)")
        
        processor_start = time.time()
        # Get device from model to ensure processor uses the same device (CPU or CUDA)
        device = next(model.parameters()).device
        processor = Sam3Processor(model, device=device, confidence_threshold=request.confidence_threshold)
        inference_state = processor.set_image(image)
        logger.info(f"Initialized processor (took {time.time() - processor_start:.2f}s)")
        
        prompt_start = time.time()
        if request.prompt:
            logger.info(f"Setting text prompt: '{request.prompt}'")
            inference_state = processor.set_text_prompt(
                state=inference_state,
                prompt=request.prompt
            )
        
        if request.boxes:
            logger.info(f"Setting {len(request.boxes)} box prompt(s)")
            processor.reset_all_prompts(inference_state)
            for idx, box in enumerate(request.boxes):
                norm_box = [box.cx, box.cy, box.w, box.h]
                logger.debug(f"Box {idx}: {norm_box}, label={box.label}")
                inference_state = processor.add_geometric_prompt(
                    state=inference_state,
                    box=norm_box,
                    label=box.label
                )
        logger.info(f"Prompt processing (took {time.time() - prompt_start:.2f}s)")
        
        inference_start = time.time()
        data_list = []
        
        if isinstance(inference_state, dict) and "masks" in inference_state:
            masks = inference_state["masks"]
            scores = inference_state.get("scores")
            boxes = inference_state.get("boxes")
            
            if masks is not None:
                # Handle tensor masks
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().float().numpy()
                
                if scores is not None and isinstance(scores, torch.Tensor):
                    scores = scores.cpu().float().numpy()
                
                if boxes is not None and isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().float().numpy()
                
                logger.info(f"Masks shape: {masks.shape}")
                logger.info(f"Scores shape: {scores.shape if scores is not None else 'N/A'}")
                
                num_masks = masks.shape[0] if len(masks.shape) > 2 else 1
                
                # Limit to requested number of results
                num_masks = min(num_masks, request.n) if request.n else num_masks
                
                for i in range(num_masks):
                    if len(masks.shape) > 2:
                        mask = masks[i]
                    else:
                        mask = masks
                    
                    score = float(scores[i]) if scores is not None and len(scores) > i else 0.5
                    
                    if len(mask.shape) == 3:
                        mask_2d = mask[0] if mask.shape[0] == 1 else mask.max(axis=0)
                    else:
                        mask_2d = mask
                    
                    rows = np.any(mask_2d, axis=1)
                    cols = np.any(mask_2d, axis=0)
                    
                    if rows.any() and cols.any():
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                    else:
                        bbox = [0.0, 0.0, 0.0, 0.0]
                    
                    mask_base64 = encode_mask_to_base64(mask_2d)
                    
                    data_list.append(
                        ImageData(
                            b64_json=mask_base64,
                            revised_prompt=request.prompt,
                            score=score,
                            bbox=bbox
                        )
                    )
        
        logger.info(f"Mask extraction complete (took {time.time() - inference_start:.2f}s)")
        logger.info(f"Generated {len(data_list)} mask(s)")
        
        response = SAM3Response(
            created=int(time.time()),
            data=data_list
        )
        
        total_time = time.time() - start_time
        logger.info(f"✓ Request complete: {len(data_list)} masks in {total_time:.2f}s")
        
        return response
        
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.status_code} - {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Segmentation failed with unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
