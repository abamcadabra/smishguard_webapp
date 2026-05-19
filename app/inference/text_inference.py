from pathlib import Path
import tensorflow as tf
from transformers import TFAutoModel

from app.processing.text_preprocess import preprocess_mbert

TEXT_MODEL_DIR = Path("models/text_model")
TEXT_MODEL_PATH = TEXT_MODEL_DIR / "bert_bilstm.keras"
TEXT_THRESHOLD = 0.5
TEXT_MODEL_VERSION = "bert_bilstm_v1"


bert_model = None

def bert_layer(inputs):
    global bert_model

    if bert_model is None:
        bert_model = TFAutoModel.from_pretrained("bert-base-multilingual-cased")

    input_ids, attention_mask = inputs

    outputs = bert_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    training=False
    )   

    return outputs.last_hidden_state

class TextModelService:
    def __init__(self):
        self.model = None

    def load(self):
        self.model = tf.keras.models.load_model(
        TEXT_MODEL_PATH,
        compile=False,
        custom_objects={"bert_layer": bert_layer}
    )

    def predict(self, text: str) -> tuple[float | None, int]:
        if self.model is None:
            raise RuntimeError("Text model is not loaded")

        if text is None or not str(text).strip():
            return None, 0

        encoded = preprocess_mbert([text])

        inputs = [
            encoded["input_ids"],
            encoded["attention_mask"]
        ]

        prob = float(
        self.model(
        inputs,
        training=False
        ).numpy().ravel()[0])
        pred = 1 if prob >= TEXT_THRESHOLD else 0
        return prob, pred


text_service = TextModelService()