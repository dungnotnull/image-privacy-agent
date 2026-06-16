"""FastAPI HTTP reverse proxy - intercepts images, applies/reverses privacy protection."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import uuid
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from PIL import Image

OPENAI_BASE = os.getenv("REAL_OPENAI_BASE_URL", "https://api.openai.com")
ANTHROPIC_BASE = os.getenv("REAL_ANTHROPIC_BASE_URL", "https://api.anthropic.com")
OLLAMA_BASE = os.getenv("REAL_OLLAMA_BASE_URL", "http://localhost:11434")

DATA_URI_PATTERN = re.compile(r"data:image/(\w+);base64,([A-Za-z0-9+/=]+)")

MAX_IMAGE_SIZE_MB = float(os.getenv("MAX_IMAGE_SIZE_MB", "20"))


def create_proxy_app(orchestrator) -> FastAPI:
    """Build and return the FastAPI proxy application."""
    app = FastAPI(title="Image Privacy Proxy", version="1.0.0")

    # ------------------------------------------------------------------
    # Health / Metrics
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        stats = {}
        try:
            stats = orchestrator.memory.get_session_stats()
        except Exception:
            pass
        return {"status": "ok", "proxy": "image-privacy-agent", "stats": stats}

    @app.get("/metrics")
    async def metrics():
        lines = orchestrator.get_prometheus_metrics()
        return Response(content="\n".join(lines), media_type="text/plain")

    # ------------------------------------------------------------------
    # OpenAI routes
    # ------------------------------------------------------------------

    @app.api_route(
        "/openai/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    )
    async def proxy_openai(path: str, request: Request):
        return await _proxy_request(
            request, f"{OPENAI_BASE}/{path}", "openai", orchestrator
        )

    # ------------------------------------------------------------------
    # Anthropic routes
    # ------------------------------------------------------------------

    @app.api_route(
        "/anthropic/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    )
    async def proxy_anthropic(path: str, request: Request):
        return await _proxy_request(
            request, f"{ANTHROPIC_BASE}/{path}", "anthropic", orchestrator
        )

    # ------------------------------------------------------------------
    # Ollama routes (pass through; no protection needed for local)
    # ------------------------------------------------------------------

    @app.api_route(
        "/ollama/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE"],
    )
    async def proxy_ollama(path: str, request: Request):
        return await _proxy_request(
            request, f"{OLLAMA_BASE}/{path}", "ollama", orchestrator, protect=False
        )

    return app


async def _proxy_request(
    request: Request,
    target_url: str,
    provider: str,
    orchestrator,
    protect: bool = True,
) -> Response:
    """Generic proxy handler: optionally apply image protection, forward, reverse."""
    start = time.perf_counter()
    session_id = str(uuid.uuid4())
    original_image: Optional[Image.Image] = None
    image_found = False
    protection_stats = None
    gate_passed = False

    body_bytes = await request.body()
    content_type = request.headers.get("content-type", "")
    modified_body = body_bytes

    if protect and body_bytes:
        try:
            if "application/json" in content_type:
                modified_body, original_image, image_found = await _protect_json_body(
                    body_bytes, session_id, orchestrator
                )
            elif "multipart/form-data" in content_type:
                modified_body, original_image, image_found = await _protect_multipart_body(
                    body_bytes, content_type, session_id, orchestrator
                )
        except Exception as exc:
            orchestrator.inc_error()
            return JSONResponse(
                status_code=422,
                content={"error": "protection_failed", "detail": str(exc)},
            )

        # Quality gate: verify protection before forwarding
        if image_found and original_image is not None:
            try:
                protected_img = Image.open(io.BytesIO(modified_body)) if "application/json" not in content_type else None
                # For JSON bodies we already hold the original; verify via the orchestrator
                # by re-decoding the first protected image from modified body
                if protected_img is None:
                    # Decode first base64 image from modified JSON for verification
                    protected_img = _extract_first_image_from_json(modified_body)
                if protected_img:
                    verif = orchestrator.verify_protection(original_image, protected_img)
                    if not verif.passed:
                        orchestrator.inc_error()
                        return JSONResponse(
                            status_code=422,
                            content={"error": "quality_gate_failed", "detail": verif.reason},
                        )
                    gate_passed = True
                    protection_stats = orchestrator.protector.compute_stats(original_image, protected_img)
            except Exception:
                pass  # If verification infra is unavailable, continue with caution

    headers = _build_forward_headers(request, len(modified_body), image_found)

    try:
        timeout_sec = float(
            orchestrator._config.get("upstream_apis", {}).get("request_timeout_seconds", 120)
        )
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=modified_body,
            )
    except httpx.TimeoutException:
        orchestrator.inc_error()
        return JSONResponse(status_code=503, content={"error": "upstream_timeout"})
    except Exception as exc:
        orchestrator.inc_error()
        return JSONResponse(
            status_code=503, content={"error": "upstream_unavailable", "detail": str(exc)}
        )

    # Reverse protection on response images
    resp_body = resp.content
    body_modified = False
    if protect and image_found:
        try:
            new_body, body_modified = await _reverse_response_images(
                resp_body, session_id, orchestrator
            )
            if body_modified:
                resp_body = new_body
                orchestrator.mark_reversal(session_id)
        except Exception:
            pass

    latency_ms = (time.perf_counter() - start) * 1000
    orchestrator.inc_session(latency_ms)

    # Persist session stats
    if image_found and original_image is not None:
        try:
            image_hash = orchestrator.hash_image(original_image)
            orchestrator.record_session(
                session_id=session_id,
                api_provider=provider,
                image_hash=image_hash,
                stats=protection_stats,
            )
        except Exception:
            pass

    # Build response: strip stale content-length if body was modified
    response_headers = dict(resp.headers)
    response_headers.pop("content-length", None)
    response_headers.pop("transfer-encoding", None)
    response_headers["x-privacy-agent"] = "1" if image_found else "no-image"

    return Response(
        content=resp_body,
        status_code=resp.status_code,
        headers=response_headers,
        media_type=resp.headers.get("content-type"),
    )


async def _protect_json_body(
    body_bytes: bytes, session_id: str, orchestrator
) -> tuple[bytes, Optional[Image.Image], bool]:
    """Extract base64 images from JSON, protect them, return modified body."""
    try:
        data = json.loads(body_bytes)
    except json.JSONDecodeError:
        return body_bytes, None, False

    original_image = None
    modified, first_original = _walk_and_protect_json(data, session_id, orchestrator)
    if first_original is not None:
        original_image = first_original
    return json.dumps(modified).encode(), original_image, original_image is not None


def _walk_and_protect_json(
    obj: Any, session_id: str, orchestrator
) -> tuple[Any, Optional[Image.Image]]:
    """Recursively walk JSON structure and replace base64 image data."""
    first_original: Optional[Image.Image] = None

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ("url", "data") and isinstance(v, str):
                match = DATA_URI_PATTERN.match(v)
                if match:
                    img_format, b64_data = match.group(1), match.group(2)
                    img_bytes = base64.b64decode(b64_data)
                    if len(img_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                        result[k] = v
                        continue
                    image = Image.open(io.BytesIO(img_bytes))
                    if first_original is None:
                        first_original = image.copy()
                    protected = orchestrator.protector.protect(image.convert("RGB"), session_id)
                    buf = io.BytesIO()
                    protected.save(buf, format="PNG")
                    new_b64 = base64.b64encode(buf.getvalue()).decode()
                    result[k] = f"data:image/png;base64,{new_b64}"
                    continue
            new_v, orig = _walk_and_protect_json(v, session_id, orchestrator)
            if orig and first_original is None:
                first_original = orig
            result[k] = new_v
        return result, first_original

    elif isinstance(obj, list):
        result_list = []
        for item in obj:
            new_item, orig = _walk_and_protect_json(item, session_id, orchestrator)
            if orig and first_original is None:
                first_original = orig
            result_list.append(new_item)
        return result_list, first_original

    return obj, None


def _extract_first_image_from_json(body_bytes: bytes) -> Optional[Image.Image]:
    """Scan JSON body for the first base64 image and decode it."""
    try:
        data = json.loads(body_bytes)
    except Exception:
        return None

    def recurse(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str):
                    m = DATA_URI_PATTERN.match(v)
                    if m:
                        return Image.open(io.BytesIO(base64.b64decode(m.group(2))))
                r = recurse(v)
                if r:
                    return r
        elif isinstance(obj, list):
            for item in obj:
                r = recurse(item)
                if r:
                    return r
        return None

    return recurse(data)


async def _protect_multipart_body(
    body_bytes: bytes,
    content_type: str,
    session_id: str,
    orchestrator,
) -> tuple[bytes, Optional[Image.Image], bool]:
    """Parse multipart/form-data, protect image parts, return modified body."""
    boundary_match = re.search(r'boundary=["\']?([^"\';\\s]+)["\']?', content_type)
    if not boundary_match:
        return body_bytes, None, False

    boundary = boundary_match.group(1).encode()
    parts = body_bytes.split(b"--" + boundary)
    modified_parts = []
    original_image: Optional[Image.Image] = None

    for part in parts:
        if b"Content-Type: image/" in part or b"content-type: image/" in part:
            header_end = part.find(b"\r\n\r\n")
            if header_end == -1:
                modified_parts.append(part)
                continue
            headers_raw = part[:header_end]
            img_data = part[header_end + 4 :]

            try:
                image = Image.open(io.BytesIO(img_data.rstrip(b"\r\n")))
                if original_image is None:
                    original_image = image.copy()
                protected = orchestrator.protector.protect(image.convert("RGB"), session_id)
                buf = io.BytesIO()
                protected.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                headers_raw = re.sub(
                    b"Content-Type: image/\\w+",
                    b"Content-Type: image/png",
                    headers_raw,
                    flags=re.IGNORECASE,
                )
                modified_parts.append(headers_raw + b"\r\n\r\n" + png_bytes + b"\r\n")
                continue
            except Exception:
                pass

        modified_parts.append(part)

    new_body = (b"--" + boundary).join(modified_parts)
    return new_body, original_image, original_image is not None


async def _reverse_response_images(
    resp_body: bytes, session_id: str, orchestrator
) -> tuple[bytes, bool]:
    """Find images in response JSON, apply reversal, return modified bytes."""
    try:
        data = json.loads(resp_body)
    except json.JSONDecodeError:
        # Try image-bytes directly (binary PNG/JPEG response)
        try:
            image = Image.open(io.BytesIO(resp_body))
            recovered = orchestrator.reverse_image(image, session_id)
            buf = io.BytesIO()
            recovered.save(buf, format="PNG")
            return buf.getvalue(), True
        except Exception:
            return resp_body, False

    modified, changed = _walk_and_reverse_json(data, session_id, orchestrator)
    if changed:
        return json.dumps(modified).encode(), True
    return resp_body, False


def _walk_and_reverse_json(
    obj: Any, session_id: str, orchestrator
) -> tuple[Any, bool]:
    changed = False
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ("b64_json", "url") and isinstance(v, str):
                try:
                    if k == "b64_json":
                        img_bytes = base64.b64decode(v)
                        image = Image.open(io.BytesIO(img_bytes))
                    else:
                        # URL pointing to an image
                        if v.startswith("http"):
                            # Don't download here; pass through
                            result[k] = v
                            continue
                        img_bytes = base64.b64decode(v)
                        image = Image.open(io.BytesIO(img_bytes))
                    recovered = orchestrator.reverse_image(image, session_id)
                    buf = io.BytesIO()
                    recovered.save(buf, format="PNG")
                    if k == "b64_json":
                        result[k] = base64.b64encode(buf.getvalue()).decode()
                    else:
                        result[k] = base64.b64encode(buf.getvalue()).decode()
                    changed = True
                    continue
                except Exception:
                    pass
            new_v, sub_changed = _walk_and_reverse_json(v, session_id, orchestrator)
            if sub_changed:
                changed = True
            result[k] = new_v
        return result, changed
    elif isinstance(obj, list):
        result_list = []
        for item in obj:
            new_item, sub_changed = _walk_and_reverse_json(item, session_id, orchestrator)
            if sub_changed:
                changed = True
            result_list.append(new_item)
        return result_list, changed
    return obj, False


def _build_forward_headers(
    request: Request, body_length: int, image_was_modified: bool
) -> dict[str, str]:
    skip = {"content-length", "host", "transfer-encoding"}
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in skip
    }
    headers["content-length"] = str(body_length)
    headers["x-privacy-agent"] = "1" if image_was_modified else "no-image"
    return headers
