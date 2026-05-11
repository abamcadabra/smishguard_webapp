import re
import uuid
from datetime import datetime, timezone


URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}\S*)"
)


def generate_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:12]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_url(text):
    m = URL_PATTERN.search(text)
    return m.group(0) if m else None

def remove_url(text):
    return URL_PATTERN.sub("", text)