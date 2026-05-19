from pathlib import Path

import tensorflow as tf
from transformers import TFAutoModel

from app.processing.text_preprocess import preprocess_mbert

# =========================================================
# Config
# =========================================================

TEXT_MODEL_DIR = Path("models/text_model")

TEXT_MODEL_PATH = TEXT_MODEL_DIR / "bert_bilstm.keras"

TEXT_THRESHOLD = 0.5
TEXT_MODEL_VERSION = "bert_bilstm_v1"

MODEL_NAME = "bert-base-multilingual-cased"

# =========================================================
# Global BERT
# =========================================================

bert_model = None


def bert_layer(inputs):

    global bert_model

    # load BERT once
    if bert_model is None:
        bert_model = TFAutoModel.from_pretrained(
            MODEL_NAME,
            from_pt=False,
            use_safetensors=False
        )

        bert_model.trainable = False

    input_ids, attention_mask = inputs

    outputs = bert_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        training=False
    )

    return outputs.last_hidden_state


# =========================================================
# Service
# =========================================================

class TextModelService:

    def __init__(self):
        self.model = None

    def load(self):

        self.model = tf.keras.models.load_model(
            TEXT_MODEL_PATH,
            custom_objects={
                "bert_layer": bert_layer
            },
            compile=False
        )

    def predict(self, text: str):

        if self.model is None:
            raise RuntimeError("Text model is not loaded")

        if text is None or not str(text).strip():
            return None, 0

        # preprocess + tokenize
        encoded = preprocess_mbert([text])

        # IMPORTANT:
        # model เดิม train แบบ list input
        inputs = [
            encoded["input_ids"],
            encoded["attention_mask"]
        ]

        # deterministic inference
        prob = float(
            self.model(
                inputs,
                training=False
            ).numpy().ravel()[0]
        )

        pred = 1 if prob >= TEXT_THRESHOLD else 0

        return prob, pred


# singleton
text_service = TextModelService()