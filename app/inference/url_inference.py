from pathlib import Path
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.processing.url_preprocess import normalize_url_soft, url_to_tokens_rb_for_model

URL_MODEL_DIR = Path("models/url_model")
URL_MODEL_PATH = URL_MODEL_DIR / "url_bilstm_fasttext.keras"
WORD2IDX_PATH = URL_MODEL_DIR / "word2idx.pkl"
MAXLEN_PATH = URL_MODEL_DIR / "maxlen.pkl"
THRESHOLD_PATH = URL_MODEL_DIR / "threshold.pkl"
URL_MODEL_VERSION = "url_bilstm_fasttext_v1"


class URLModelService:
    def __init__(self):
        self.model = None
        self.word2idx = None
        self.maxlen = None
        self.threshold = 0.5

    def load(self):
        self.model = tf.keras.models.load_model(URL_MODEL_PATH, compile=False)
        self.word2idx = joblib.load(WORD2IDX_PATH)
        self.maxlen = joblib.load(MAXLEN_PATH)

        if THRESHOLD_PATH.exists():
            self.threshold = float(joblib.load(THRESHOLD_PATH))

    def _texts_to_sequences(self, token_lists: list[list[str]]) -> list[list[int]]:
        oov_idx = self.word2idx.get("<OOV>", 1)
        return [
            [self.word2idx.get(tok, oov_idx) for tok in tokens]
            for tokens in token_lists
        ]

    def predict(self, raw_url: str | None) -> tuple[str | None, float | None, int]:
        if self.model is None or self.word2idx is None or self.maxlen is None:
            raise RuntimeError("URL model is not loaded")

        normalized_url = normalize_url_soft(raw_url)
        if not normalized_url:
            return None, None, 0

        tokens = url_to_tokens_rb_for_model(normalized_url)
        if not tokens:
            return normalized_url, None, 0

        seqs = self._texts_to_sequences([tokens])
        x = pad_sequences(seqs, maxlen=self.maxlen, padding="post", truncating="post")

        prob = float(self.model.predict(x, verbose=0).ravel()[0])
        pred = 1 if prob >= self.threshold else 0

        return normalized_url, prob, pred


url_service = URLModelService()