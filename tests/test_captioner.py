"""Unit tests for BLIP captioning and caption cleaning."""

import math
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
import torch

from vidcap.captioner import BLIPCaptioner, clean_blip_caption
from vidcap.config import Config


def test_clean_blip_caption_arafed():
    # Test removal of "arafed" artifact
    assert clean_blip_caption("arafed a man walking on a street") == "A man walking on a street"
    assert clean_blip_caption("a photo of arafed dog in park") == "A photo of dog in park"
    assert clean_blip_caption("arafed") == ""
    assert clean_blip_caption("arafed arafed") == ""


def test_clean_blip_caption_duplicates():
    # Test consecutive duplicate word removal
    assert clean_blip_caption("a man walking the the street") == "A man walking the street"
    # Test consecutive duplicate phrase removal
    assert clean_blip_caption("a cat on a table a cat on a table") == "A cat on a table"


def test_clean_blip_caption_whitespace_and_punctuation():
    assert clean_blip_caption("  ... hello world ,,, ") == "Hello world"
    assert clean_blip_caption("") == ""
    assert clean_blip_caption(None) == ""


@patch("vidcap.captioner.BlipForConditionalGeneration.from_pretrained")
@patch("vidcap.captioner.BlipProcessor.from_pretrained")
def test_blip_batched_generate_calls(mock_processor_cls, mock_model_cls):
    """
    Acceptance criterion #1:
    Verify that frames are processed strictly in batches and generate() is called
    ceil(N / batch_size) times.
    """
    mock_processor = MagicMock()
    mock_processor_cls.return_value = mock_processor

    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model

    # Mock tensor output
    mock_processor.return_value.to.return_value = {"input_ids": torch.zeros((1, 1))}
    mock_processor.batch_decode.side_effect = lambda outputs, **kwargs: ["A test caption"] * len(outputs)

    config = Config(blip_batch_size=4, blip_device="cpu")
    captioner = BLIPCaptioner(config)

    # Prepare 10 fake images
    images = [Image.new("RGB", (32, 32)) for _ in range(10)]

    # Mock generate to return a tensor with matching batch dimension
    def mock_generate(**kwargs):
        # Determine batch dimension from inputs
        return torch.zeros((4, 5))

    mock_model.generate.side_effect = mock_generate

    batch_size = 4
    captioner.caption_images(images, batch_size=batch_size)

    expected_calls = math.ceil(len(images) / batch_size)  # ceil(10 / 4) = 3
    assert mock_model.generate.call_count == expected_calls
