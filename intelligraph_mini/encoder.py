"""Minimal MiniLM encoder loader — bundled model, no network calls."""

import os
import sys
import logging
from pathlib import Path

log = logging.getLogger(__name__)
_ENCODER = None
_ENCODER_ERR = None

_MODEL_DIR = Path(__file__).parent / "models" / "all-MiniLM-L6-v2"


def _patch_hf_validator():
    """Patch huggingface_hub to accept local directory paths on Windows.

    sentence-transformers 3.x has a bug where it doesn't detect local
    directories properly on Windows — it tries to parse the path as a
    HuggingFace repo ID, which fails on backslash paths.
    """
    try:
        import huggingface_hub.utils._validators as hf_val
        _original = hf_val.validate_repo_id
        def _patched(repo_id):
            if os.path.isdir(str(repo_id)):
                return
            return _original(repo_id)
        hf_val.validate_repo_id = _patched
    except Exception:
        pass


def get_encoder():
    global _ENCODER, _ENCODER_ERR
    if _ENCODER is not None:
        return _ENCODER
    if _ENCODER_ERR is not None:
        return None
    try:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        _patch_hf_validator()
        from sentence_transformers import SentenceTransformer
        model_path = str(_MODEL_DIR)
        if not os.path.isdir(model_path):
            _ENCODER_ERR = f"Model dir not found: {model_path}"
            print(f"[intelligraph-mini] {_ENCODER_ERR}")
            return None
        _ENCODER = SentenceTransformer(model_path)
        print(f"[intelligraph-mini] encoder ready (dim={_ENCODER.get_sentence_embedding_dimension()})")
        return _ENCODER
    except Exception as e:
        _ENCODER_ERR = str(e)
        print(f"[intelligraph-mini] encoder init failed: {e}")
        log.warning("Encoder init failed: %s", e)
        return None
