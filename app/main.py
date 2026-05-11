import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import PredictRequest, PredictResponse
from app.utils import utc_now_iso, extract_url
from app.processing.text_preprocess import preprocess_text
from app.inference.text_inference import text_service, TEXT_MODEL_VERSION
from app.inference.url_inference import url_service, URL_MODEL_VERSION
from app.fusion import or_fusion
from app.logging_store import save_prediction_log
from app.monitoring import (
    record_request,
    record_success,
    record_failure,
    record_reason,
    record_has_url,
    get_metrics,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    text_service.load()
    url_service.load()
    yield


app = FastAPI(
    title="SmishGuard Realtime Inference API",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "text_model_loaded": text_service.model is not None,
        "url_model_loaded": url_service.model is not None
    }


@app.get("/metrics")
def metrics():
    return get_metrics()


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    record_request()
    start = time.perf_counter()

    try:
        processed_at = utc_now_iso()

        raw_text = payload.message_text.strip()
        if not raw_text:
            raise HTTPException(status_code=400, detail="message_text must not be empty")

        # ใช้สำหรับเก็บลง log / response
        clean_text = preprocess_text(raw_text)

        # extract URL จากข้อความดิบ
        extracted_url = extract_url(raw_text)
        record_has_url(extracted_url is not None)

        # ส่ง raw_text เข้า text service โดยตรง
        # เพราะ text_service.predict() จะเรียก preprocess_mbert() ภายในเอง
        text_prob, text_pred = text_service.predict(raw_text)

        # URL branch
        normalized_url, url_prob, url_pred = url_service.predict(extracted_url)

        # OR fusion
        final_pred, reason = or_fusion(text_pred=text_pred, url_pred=url_pred)

        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        result = {
            "clean_text": clean_text,
            "extracted_url": extracted_url,
            "normalized_url": normalized_url,
            "text_prob": text_prob,
            "text_pred": text_pred,
            "url_prob": url_prob,
            "url_pred": url_pred,
            "final_pred": final_pred,
            "fusion_rule": "OR",
            "reason": reason,
            "latency_ms": latency_ms,
            "model_version_text": TEXT_MODEL_VERSION,
            "model_version_url": URL_MODEL_VERSION,
            "processed_at": processed_at,
        }

        save_prediction_log(result)
        record_reason(reason)
        record_success()

        return result

    except HTTPException:
        record_failure()
        raise
    except Exception as e:
        record_failure()
        raise HTTPException(status_code=500, detail=str(e))