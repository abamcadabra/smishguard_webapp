import re
import unicodedata
import ipaddress
import tldextract
from urllib.parse import (
    urlparse, urlunparse,
    unquote, parse_qsl, urlencode
)

TRACK_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign",
    "utm_term", "utm_content",
    "gclid", "fbclid",
    "mc_eid", "mc_cid",
    "igshid", "si", "spm"
}

SHORTENER_DOMAINS = {
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly",
    "buff.ly", "adf.ly", "is.gd", "cutt.ly",
    "rebrand.ly", "soo.gd", "shorturl.at",
    "trib.al", "rb.gy", "lnkd.in"
}

def normalize_url_soft(
    u: str,
    *,
    default_scheme: str = "http",
    allow_schemes: tuple = ("http", "https"),
    drop_fragment: bool = True,
    drop_default_ports: bool = True,
    strip_www: bool = False,
    sort_query: bool = True,
    track_params: set = TRACK_PARAMS,
    max_url_len: int = 4096
) -> str:
    if u is None:
        return ""

    if not isinstance(u, str):
        u = str(u)

    u = unicodedata.normalize("NFKC", u.strip())
    if not u:
        return ""

    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.\-]*://', u):
        u = f"{default_scheme}://{u}"

    if len(u) > max_url_len:
        return ""

    try:
        p = urlparse(u)

        scheme = (p.scheme or default_scheme).lower()
        if scheme not in allow_schemes:
            scheme = default_scheme

        host = p.hostname or ""
        try:
            host = host.encode("idna").decode("ascii").lower()
        except Exception:
            host = host.lower()

        if strip_www and host.startswith("www."):
            host = host[4:]

        port = p.port
        if drop_default_ports:
            if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
                port = None

        path = unquote(p.path or "")
        query = unquote(p.query or "")

        if query:
            pairs = parse_qsl(query, keep_blank_values=True)
            if track_params:
                pairs = [(k, v) for (k, v) in pairs if k not in track_params]
            if sort_query:
                pairs.sort()
            query = urlencode(pairs, doseq=True)

        fragment = "" if drop_fragment else p.fragment
        netloc = host + (f":{port}" if port else "")

        return urlunparse((scheme, netloc, path, p.params, query, fragment))

    except Exception:
        return ""


def _is_shortened_url(host: str) -> bool:
    try:
        ext = tldextract.extract(host)
        domain = f"{ext.domain}.{ext.suffix}".strip(".")
        return domain in SHORTENER_DOMAINS
    except Exception:
        return False

def _is_ip_host(host: str) -> bool:
    host = (host or "").strip("[]")
    try:
        ipaddress.ip_address(host)
        return True
    except Exception:
        return False

def _is_numeric_only_host(host: str) -> bool:
    try:
        ext = tldextract.extract(host)
        return bool(ext.domain) and ext.domain.isdigit()
    except Exception:
        return False

def _too_many_hexish(path_and_query: str, max_hex_len: int = 32, max_hex_tokens: int = 12) -> bool:
    toks = re.findall(r'[A-Fa-f0-9]{%d,}' % max_hex_len, path_and_query or "")
    return len(toks) >= max_hex_tokens

def _overlong_pq(path_and_query: str, max_len: int = 2048) -> bool:
    return len(path_and_query or "") > max_len

SEP_HOST = re.compile(r'\.+|-+|_+')
SEP_BODY = re.compile(r'[^A-Za-z0-9]+')

def url_to_tokens_rb_for_model(u: str):
    p = urlparse(u)

    host = p.hostname or ""
    pathq = unquote(p.path or "")
    if p.query:
        pathq += "?" + unquote(p.query)

    host_parts = [
        t for t in SEP_HOST.split(host)
        if t and t.lower() != "www"
    ]
    body_parts = [
        t for t in SEP_BODY.split(pathq)
        if t
    ]

    return ["<HOST>"] + host_parts + ["<PATH>"] + body_parts
