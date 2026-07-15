"""Minimal MiniLM encoder loader — bundled model, no network calls."""

import os
import logging

log = logging.getLogger(__name__)
_ENCODER = None
_ENCODER_ERR = None

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "all-MiniLM-L6-v2")


def get_encoder():
    global _ENCODER, _ENCODER_ERR
    if _ENCODER is not None:
        return _ENCODER
    if _ENCODER_ERR is not None:
        return None
    try:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from sentence_transformers import SentenceTransformer
        if not os.path.isdir(_MODEL_DIR):
            _ENCODER_ERR = f"Model dir not found: {_MODEL_DIR}"
            print(f"[intelligraph-mini] {_ENCODER_ERR}")
            return None
        _ENCODER = SentenceTransformer(_MODEL_DIR)
        print(f"[intelligraph-mini] encoder ready (dim={_ENCODER.get_sentence_embedding_dimension()})")
        return _ENCODER
    except Exception as e:
        _ENCODER_ERR = str(e)
        print(f"[intelligraph-mini] encoder init failed: {e}")
        return None
