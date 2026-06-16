# Image Privacy Agent — Full Technical Specification

## Executive Summary

Personal images uploaded to commercial LLM APIs (ChatGPT image analysis, Claude vision, DALL-E edits) travel unencrypted at the application layer: API providers log request bodies, the images appear in model training pipelines, and any TLS-terminating proxy or compromised CDN node can read them. The **Image Privacy Agent** runs as a localhost HTTP reverse proxy. It intercepts every outgoing image byte, applies a session-keyed reversible adversarial noise transform, and forwards the protected image to the real LLM API. On the response path, it reverses the transform so the user receives the clean edited result. The noise is tuned to be imperceptible to humans (L∞ ≤ 4/255) yet effective against pixel-level interception. A CLIP-based semantic verifier ensures the LLM still "sees" the correct semantic content.

---

## Target Users & Use Cases

| User | Trigger | Agent Behavior |
|------|---------|----------------|
| Privacy-conscious individual | Uploads headshot for LinkedIn background removal | Proxy applies noise → forwards → removes noise from returned image |
| Developer testing vision API | Sends customer images to GPT-4o for OCR | All images protected in transit; original pixel data never leaves the machine |
| Enterprise user | Uploads product photos to Claude for catalog copy generation | Noise added; LLM still generates accurate descriptions; raw pixels protected |
| Security researcher | Wants threat report for an uploaded image | ThreatAnalyzer identifies PII (faces, text, EXIF GPS) and recommends protection level |

---

## Agent Architecture

```
User App / Browser
        │
        │  HTTP to localhost:8003
        │  (configure OPENAI_BASE_URL=http://localhost:8003/openai
        │   configure ANTHROPIC_BASE_URL=http://localhost:8003/anthropic)
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ProxyInterceptor  (FastAPI, port 8003)                             │
│                                                                     │
│  1. Parse request → extract image bytes (multipart or base64)       │
│  2. Generate session_id (UUID4) + derive session_key (PBKDF2)       │
│  3. ─► AdversarialProtector.protect(image, session_id, key)         │
│         └─ pixel-space keyed noise (L∞ ≤ 4/255)                    │
│         └─ DCT block masking (8×8 blocks, high-freq coefficients)   │
│         └─ store noise_mask in MemoryManager                        │
│  4. ─► SemanticVerifier.verify(original, protected)                 │
│         └─ CLIP cosine ≥ 0.92 gate                                  │
│         └─ SSIM ≥ 0.95 gate                                         │
│         └─ block send if gate fails                                 │
│  5. Replace image bytes in request body                             │
│  6. Forward modified request → Real API (httpx async)               │
│  7. ─► Parse API response for returned images                       │
│  8. ─► AdversarialProtector.reverse(response_image, session_id)     │
│  9. Return clean response to caller                                 │
└─────────────────────────────────────────────────────────────────────┘
        │                               │
        ▼                               ▼
MemoryManager (SQLite)         ThreatAnalyzer (on-demand)
session_id → noise_mask        PII detection + LLM threat report
        │
        ▼
LLM API (OpenAI / Anthropic / Ollama)
```

---

## Full Module Catalog

### `agent/modules/adversarial_protector.py`

**Responsibility:** Core privacy transform engine. Applies and reverses keyed adversarial noise.

**Inputs:**
- `image: PIL.Image` — image to protect or recover
- `session_id: str` — unique session identifier
- `secret_key: str` — session-specific derived secret (from MemoryManager)

**Outputs:**
- Protected or recovered `PIL.Image`
- `ProtectionStats` dataclass: l_inf_norm, psnr_db, ssim, dct_energy_delta

**Noise strategy:**
1. **Pixel-space L∞ noise**: `noise = keyed_rng.integers(-ε_int, ε_int+1, shape)` where ε = 4/255 × 255 = 4 gray levels. Deterministic from session key via PCG64 PRNG seeded with PBKDF2(session_id + master_secret, salt, 100_000, dklen=32).
2. **DCT block masking** (optional, configurable): Apply 8×8 block DCT → zero out top-k highest-frequency coefficients (default k=3) → IDCT. This disrupts Fourier-domain pixel extraction while preserving spatial structure.
3. **Reversal**: `recovered = clip(protected - noise, 0, 255)` for pixel noise; IDCT with stored zeroed coefficients for DCT mode.

**Quality gate:** Called by SemanticVerifier. PSNR must be ≥ 35 dB for the protection to be considered imperceptible.

**Tools called:** None (pure NumPy/SciPy)

---

### `agent/modules/proxy_interceptor.py`

**Responsibility:** FastAPI HTTP reverse proxy. Parses API request/response bodies, orchestrates protect/forward/reverse cycle.

**Inputs:** Raw HTTP requests from user's LLM client

**Outputs:** Modified HTTP responses with noise-reversed images

**Supported API formats:**
- `POST /openai/v1/chat/completions` — OpenAI chat with vision (base64 data URIs in `messages[].content[].image_url.url`)
- `POST /openai/v1/images/edits` — DALL-E inpainting (multipart: `image`, `mask` files)
- `POST /openai/v1/images/generations` — Text-to-image (no input image; skip protection)
- `POST /anthropic/v1/messages` — Anthropic messages with vision (base64 in `content[].source.data`)
- `GET /health` — liveness probe

**Image extraction logic:**
- JSON body: recurse through dict/list; find keys matching `image_url`, `url`, `data` with value starting `data:image/`
- Multipart: find parts with `Content-Type: image/*`
- Handle PNG, JPEG, WebP, BMP inputs; always convert to PNG for lossless protection

**Forward logic:** `httpx.AsyncClient` with 120s timeout. Pass through all original headers except `Content-Length` (recalculated). Inject `X-Privacy-Agent: 1` header.

**Response parsing:** For image edit responses, extract `data[].b64_json` or download `data[].url` → apply reversal → re-encode → replace in response JSON.

---

### `agent/modules/semantic_verifier.py`

**Responsibility:** Quality gate: verify that the protected image preserves semantic content for the LLM.

**Inputs:**
- `original: PIL.Image`
- `protected: PIL.Image`
- `thresholds: dict` — configurable from agent_config.yaml

**Outputs:**
- `VerificationResult` dataclass: clip_cosine, ssim, psnr_db, passed, reason

**CLIP check:**
1. Encode both images with `openai/clip-vit-large-patch14`
2. Compute cosine similarity of image embeddings
3. Gate: cosine ≥ 0.92 (configurable)

**SSIM check:**
- Compute SSIM using `skimage.metrics.structural_similarity` on grayscale
- Gate: SSIM ≥ 0.95

**Recovery check (post-edit):**
- Compare recovered image vs reference original on unedited regions
- Gate: mean absolute error ≤ 2 gray levels on unedited mask

**Fallback:** If CLIP model unavailable (no GPU, no HF token), use heuristic PSNR ≥ 35 dB as proxy for semantic preservation.

---

### `agent/modules/threat_analyzer.py`

**Responsibility:** LLM-powered privacy threat modeling for a given image.

**Inputs:**
- `image: PIL.Image` — the image to analyze (local, never sent externally in this module)
- `context: str` — user description of how they plan to use the image

**Outputs:**
- `ThreatReport` dataclass: pii_types, risk_level (low/medium/high/critical), recommended_protection, attack_scenarios, llm_recommendations, confidence

**PII detection pipeline:**
1. Heuristic scan: detect faces (OpenCV haarcascade), text regions (ViT attention maps), geotag in EXIF metadata
2. LLM prompt: "Describe what private information is visible in this image description. List: faces, text, location indicators, identifying features." (IMAGE IS DESCRIBED BY USER — not forwarded to LLM in plaintext)
3. Risk scoring: face = high, text with PII = critical, location EXIF = high, general scene = low
4. Recommend protection level: ε=2/255 (low), ε=4/255 (medium), ε=8/255 (high)

**Attack scenarios generated by LLM:**
- Model inversion attack (reconstruct training data from model weights)
- API log scraping (provider stores raw request bytes)
- TLS man-in-the-middle (corporate proxy, malicious WiFi)
- Data breach at provider (historical precedent: multiple LLM providers breached)

**Safety gate:** ThreatAnalyzer never forwards the image to the LLM. Only the IMAGE DESCRIPTION (generated locally by user or ViT caption) is sent to the LLM API.

---

## HuggingFace Model Selection

| Model | Task | Benchmark | Reason over alternatives |
|-------|------|-----------|--------------------------|
| `openai/clip-vit-large-patch14` | Semantic embedding for quality gate | ImageNet 75.5% zero-shot; COCO I→T R@1 = 59.0% | Industry standard for image-text alignment; pip-installable |
| `google/vit-base-patch16-224` | Image quality classification | ImageNet 81.8% top-1 | Faster than ViT-L; sufficient for quality scoring |
| `BAAI/bge-large-en-v1.5` | Knowledge retrieval from SECOND-KNOWLEDGE-BRAIN.md | MTEB overall 64.23 | Consistent with cluster tools; avoids duplicate model downloads |

---

## LLM API Integration Spec

**Provider priority:** claude-opus-4-8 → gpt-4o → ollama/llama3

**Use cases and prompt budgets:**

| Use case | Prompt template | Estimated tokens |
|----------|----------------|-----------------|
| Privacy threat report | `THREAT_ANALYSIS_PROMPT` | 512 in / 1024 out |
| Knowledge synthesis | `RESEARCH_SYNTHESIS_PROMPT` | 4096 in / 2048 out |
| Protection recommendation | `PROTECTION_ADVISORY_PROMPT` | 256 in / 512 out |

**Streaming:** All LLM calls use streaming for threat reports to improve UX.

**Cost guard:** Max $0.50 per session (configurable). Reject if budget exceeded.

---

## E2E Execution Flow

1. User launches proxy: `python agent/main.py serve --port 8003`
2. User configures their LLM client: `OPENAI_BASE_URL=http://localhost:8003/openai`
3. User sends ChatGPT request with personal photo
4. ProxyInterceptor receives request at `POST /openai/v1/chat/completions`
5. Extract base64 image from `messages[0].content[1].image_url.url`
6. Decode → PIL.Image
7. Generate `session_id = uuid4()`, derive `session_key = PBKDF2(session_id + MASTER_SECRET)`
8. `AdversarialProtector.protect(image, session_id, session_key)` → protected_image
9. `SemanticVerifier.verify(image, protected_image)` → CLIP cosine = 0.96 ✓, SSIM = 0.97 ✓
10. Re-encode protected_image as base64 PNG, replace in request body
11. `httpx.AsyncClient.post(OPENAI_BASE_URL, ...)` — forward modified request
12. Receive response: `{"choices": [{"message": {"content": "I can see a person in..."}}]}`
13. Scan response body for image data — none found (text response)
14. Return response unchanged to caller
15. MemoryManager records: session_id, timestamp, image_hash, protection_stats, api_provider, cost

**Error paths:**
- SSIM < 0.95 at step 9: REJECT, log warning, suggest reducing ε in config
- API unreachable at step 11: Return 503 with `{"error": "upstream_unavailable"}`
- No image in request: Pass through transparently with `X-Privacy-Agent: no-image` header

---

## SECOND-KNOWLEDGE-BRAIN.md Integration

- **Sources:** ArXiv cs.CR + cs.CV, Semantic Scholar adversarial ML queries, IEEE S&P, USENIX Security
- **Crawl config:** Weekly Sunday 02:00; top-20 papers scored by recency (last 90 days) × relevance
- **Dedup strategy:** SHA256 of (title + first author + year); skip if already in knowledge_hashes table
- **Usage in agent:** ThreatAnalyzer retrieves relevant papers via BGE-large FAISS search → feeds to LLM for evidence-based recommendations

---

## Quality Gates (Pre-delivery)

1. `protection_ssim ≥ 0.95` — noise imperceptible to human eye
2. `clip_cosine ≥ 0.92` — LLM semantic understanding preserved
3. `recovery_ssim ≥ 0.99` on unedited regions — lossless reversal
4. `psnr_protected ≥ 35 dB` — well above perceptual threshold
5. `session_key_entropy ≥ 256 bits` — cryptographically secure session isolation
6. `proxy_latency_p95 ≤ 200ms` overhead on top of API latency
7. `memory_no_plaintext` — only noise_mask stored, never raw image bytes

---

## Test Scenarios

See `tests/test-scenarios.md` for 8 complete end-to-end scenarios.

---

## Key Design Decisions

1. **Pixel-space noise over GAN-based protection**: GAN-based adversarial examples require gradient computation and a surrogate model. Keyed L∞ noise is deterministic, reversible, and requires no model inference for the protect/reverse cycle. CLIP is only used for the quality gate, not for generating the perturbation.

2. **Local proxy over browser extension or OS-level hook**: A FastAPI proxy is language-agnostic (works with any LLM client), does not require browser permissions, and can be audited. OS-level TLS hooks (mitmproxy CA certificate) were rejected because they break certificate pinning in some apps.

3. **Session-keyed noise over fixed key**: Fixed keys are vulnerable to known-plaintext attacks. Each session uses a fresh PBKDF2-derived key. The master secret is stored locally (never transmitted).

4. **PNG lossless encoding for protected images**: JPEG compression would destroy the exact pixel-space noise values, making reversal lossy. We always re-encode to PNG before sending. API providers accept PNG for all vision endpoints.

5. **ThreatAnalyzer uses local ViT description, not raw image forwarding**: Sending the original image to an LLM to analyze its privacy risk would itself be a privacy violation. We describe the image locally and send the DESCRIPTION to the LLM.

6. **DCT masking as optional second layer**: Frequency-domain attacks (DFT of the JPEG DCT coefficients, steganographic recovery) are defended by zeroing high-frequency DCT coefficients before sending. This is reversible because we store which coefficients were zeroed.

7. **No key material leaves the host**: The master secret and all session keys live in `~/.config/image-privacy-agent/` with file permissions 0600. The proxy never logs or transmits key material.
