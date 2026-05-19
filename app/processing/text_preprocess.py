# =========================================================
# text_preprocess.py
# Preprocessing for mBERT (Smishing Detection)
# EXACTLY aligned with training pipeline
# =========================================================

import re
import pandas as pd
from typing import List

from pythainlp.util import normalize as th_normalize
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------
# Config (MUST match training)
# ---------------------------------------------------------
MODEL_NAME   = "bert-base-multilingual-cased"
MAX_LEN_BERT = 128

# ---------------------------------------------------------
# URL regex (same as training)
# ---------------------------------------------------------
URL_RE = re.compile(r"""
\b(
    (?:https?|ftp)://[^\s<>"'(){}[\]]+
  | www\.[^\s<>"'(){}[\]]+
  | (?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/[^\s<>"'(){}[\]]*)?
  | (?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+
    (?:[a-z]{2,63})
    (?::\d+)?
    (?:/[^\s<>"'(){}[\]]*)?
)
""", re.IGNORECASE | re.VERBOSE)

def remove_urls(text: str) -> str:
    return URL_RE.sub("", text)

# ---------------------------------------------------------
# Text normalization (same logic as training)
# ---------------------------------------------------------
def normalize_en(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = remove_urls(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_th(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = th_normalize(text)
    text = remove_urls(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------
# BERT tokenizer (singleton)
# ---------------------------------------------------------
_tokenizer = None

def get_tokenizer():
    global _tokenizer

    if _tokenizer is None:
        _tokenizer = PreTrainedTokenizerFast(
            tokenizer_file="models/text_model/bert_bilstm_tokenizer/tokenizer.json"
        )

    return _tokenizer

# ---------------------------------------------------------
# Encoding for mBERT
# ---------------------------------------------------------
def encode_bert(texts: List[str]):
    """
    Encode raw texts for mBERT + BiLSTM
    NOTE:
    - Do NOT fit tokenizer
    - Do NOT change max_length
    """
    tokenizer = get_tokenizer()
    return tokenizer(
        [str(t) for t in texts],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN_BERT,
        return_tensors="tf"
    )

# ---------------------------------------------------------
# High-level inference helper
# ---------------------------------------------------------
def preprocess_mbert(texts: List[str]):

    cleaned = []

    for text in texts:

        if re.search(r"[\u0E00-\u0E7F]", str(text)):
            cleaned.append(normalize_th(text))
        else:
            cleaned.append(normalize_en(text))

    return encode_bert(cleaned)

def preprocess_text(text: str) -> str:
    """
    Lightweight preprocessing for logging / display
    (NOT used for model inference)
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # ถ้ามีภาษาไทย → ใช้ TH pipeline
    if re.search(r"[\u0E00-\u0E7F]", text):
        return normalize_th(text)
    else:
        return normalize_en(text)