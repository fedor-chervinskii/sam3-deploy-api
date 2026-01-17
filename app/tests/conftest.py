"""Test configuration and fixtures for SAM3 API tests."""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app as fastapi_app


@pytest.fixture(scope="session")
def client():
    """Create a test client that properly initializes batch_processor via lifespan.
    
    Uses session scope to load the model only once for all tests, improving CI performance.
    The lifespan will initialize batch_processor, executor, and load the model on first test.
    """
    with TestClient(fastapi_app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def test_image_base64() -> str:
    """Create a simple test image and return as base64."""
    # Create a simple RGB image (100x100 red square)
    img = Image.new('RGB', (100, 100), color='red')
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str


@pytest.fixture(scope="session")
def test_image_with_data_uri(test_image_base64: str) -> str:
    """Return test image with data URI prefix."""
    return f"data:image/png;base64,{test_image_base64}"


@pytest.fixture(scope="session")
def sample_text_request(test_image_base64: str) -> dict:
    """Sample request with text prompt."""
    return {
        "image": test_image_base64,
        "prompt": "object",
        "confidence_threshold": 0.5
    }


@pytest.fixture(scope="session")
def sample_box_request(test_image_base64: str) -> dict:
    """Sample request with box prompts."""
    return {
        "image": test_image_base64,
        "boxes": [
            {
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.3,
                "h": 0.3,
                "label": True
            }
        ],
        "confidence_threshold": 0.5
    }


@pytest.fixture(scope="session")
def sample_combined_request(test_image_base64: str) -> dict:
    """Sample request with both text and box prompts."""
    return {
        "image": test_image_base64,
        "prompt": "object",
        "boxes": [
            {
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.3,
                "h": 0.3,
                "label": True
            }
        ],
        "confidence_threshold": 0.3
    }
