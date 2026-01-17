"""FastAPI application for serving SAM3 model."""

import asyncio
import base64
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
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


# Global model instance and thread pool executor
model_state: Dict = {}
executor: ThreadPoolExecutor = None


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
    global executor
    
    # Startup: load model and initialize executor
    logger.info("Loading SAM3 model...")
    model_state["model"] = load_model()
    logger.info("SAM3 model loaded successfully!")
    
    # Initialize thread pool executor for CPU-bound operations
    # Use 4 workers to handle multiple concurrent requests
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sam3-worker")
    logger.info("Thread pool executor initialized with 4 workers")
    
    yield
    
    # Shutdown: cleanup
    logger.info("Shutting down executor...")
    executor.shutdown(wait=True)
    model_state.clear()
    logger.info("Cleanup complete")


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


async def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image asynchronously."""
    def _decode():
        try:
            # Remove data URI prefix if present
            img_str = base64_string
            if "," in img_str:
                img_str = img_str.split(",", 1)[1]
            
            image_bytes = base64.b64decode(img_str)
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _decode)


async def encode_mask_to_base64(mask: np.ndarray) -> str:
    """Encode binary mask to base64 PNG string asynchronously."""
    def _encode():
        # Convert boolean mask to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Encode as PNG
        success, buffer = cv2.imencode(".png", mask_uint8)
        if not success:
            raise ValueError("Failed to encode mask as PNG")
        
        # Convert to base64
        base64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
        return base64_str
    
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _encode)


async def run_segmentation(model, image: Image.Image, request: SAM3Request) -> list:
    """Run segmentation with support for multiple prompts asynchronously.
    
    Args:
        model: The SAM3 model
        image: PIL Image to segment
        request: SAM3Request with prompts and parameters
        
    Returns:
        List of ImageData objects containing masks
    """
    # Prepare processor and inference state in thread pool
    async def prepare_inference_state():
        def _prepare():
            device = next(model.parameters()).device
            processor = Sam3Processor(model, device=device, confidence_threshold=request.confidence_threshold)
            inference_state = processor.set_image(image)
            return processor, inference_state
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, _prepare)
    
    processor, inference_state = await prepare_inference_state()
    
    data_list = []
    
    # Parse text prompts (support comma-separated multiple prompts)
    text_prompts = []
    if request.prompt:
        text_prompts = [p.strip() for p in request.prompt.split(',') if p.strip()]
    
    # Process text prompts asynchronously in parallel
    if text_prompts:
        async def process_single_text_prompt(prompt_text):
            def _process():
                temp_state = dict(inference_state)
                processor.reset_all_prompts(temp_state)
                
                temp_state = processor.set_text_prompt(
                    state=temp_state,
                    prompt=prompt_text
                )
                return temp_state, prompt_text
            
            loop = asyncio.get_event_loop()
            temp_state, prompt = await loop.run_in_executor(executor, _process)
            
            # Extract and encode masks asynchronously
            masks_for_prompt = await extract_masks_from_state_async(
                temp_state, 
                prompt, 
                request.n
            )
            return masks_for_prompt
        
        # Process all text prompts concurrently
        results = await asyncio.gather(*[process_single_text_prompt(p) for p in text_prompts])
        for masks in results:
            data_list.extend(masks)
    
    # Process box prompts asynchronously
    if request.boxes:
        async def process_box_prompts():
            def _process():
                state = dict(inference_state)
                processor.reset_all_prompts(state)
                for idx, box in enumerate(request.boxes):
                    norm_box = [box.cx, box.cy, box.w, box.h]
                    state = processor.add_geometric_prompt(
                        state=state,
                        box=norm_box,
                        label=box.label
                    )
                return state
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(executor, _process)
        
        box_state = await process_box_prompts()
        
        # Extract and encode masks asynchronously
        masks_for_boxes = await extract_masks_from_state_async(
            box_state, 
            None,
            None
        )
        data_list.extend(masks_for_boxes)
    
    return data_list


async def extract_masks_from_state_async(state: dict, prompt: str, max_masks: int = None) -> list:
    """Extract masks from inference state and convert to ImageData objects asynchronously.
    
    Args:
        state: Inference state dictionary containing masks, scores, and boxes
        prompt: Text prompt associated with these masks (can be None for box prompts)
        max_masks: Maximum number of masks to return (None = return all)
        
    Returns:
        List of ImageData objects with associated prompt information
    """
    data_list = []
    
    if not isinstance(state, dict) or "masks" not in state:
        return data_list
    
    masks = state["masks"]
    scores = state.get("scores")
    boxes = state.get("boxes")
    
    if masks is None:
        return data_list
    
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().float().numpy()
    
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().float().numpy()
    
    if boxes is not None and isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().float().numpy()
    
    logger.info(f"Masks shape: {masks.shape}")
    logger.info(f"Scores shape: {scores.shape if scores is not None else 'N/A'}")
    
    num_masks = masks.shape[0] if len(masks.shape) > 2 else 1
    num_masks = min(num_masks, max_masks) if max_masks else num_masks
    
    # Process masks in parallel
    async def process_single_mask(i):
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
        
        # Async mask encoding
        mask_base64 = await encode_mask_to_base64(mask_2d)
        
        return ImageData(
            b64_json=mask_base64,
            prompt=prompt,
            score=score,
            bbox=bbox
        )
    
    # Process all masks concurrently
    data_list = await asyncio.gather(*[process_single_mask(i) for i in range(num_masks)])
    
    return data_list


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
    - Multiple comma-separated prompts: "car, person, dog" for multiple classes
    - Visual prompts: Provide bounding boxes as examples
    - Multiple results: Use 'n' parameter to get multiple mask variations
    
    Processes requests asynchronously for efficient handling of concurrent requests.
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
        # Async image decoding
        image = await decode_base64_image(request.image)
        width, height = image.size
        logger.info(f"Decoded image: {width}x{height} pixels (took {time.time() - decode_start:.2f}s)")
        
        # Run segmentation asynchronously
        inference_start = time.time()
        data_list = await run_segmentation(model, image, request)
        logger.info(f"Inference complete (took {time.time() - inference_start:.2f}s)")
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
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
