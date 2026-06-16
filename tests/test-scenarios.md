# Image Privacy Agent — Test Scenarios

## Scenario 1: Golden Path — Protect Personal Photo for GPT-4o Analysis

**Setup:** User uploads a personal headshot to ChatGPT (configured to use local proxy).

**Input:** 512×512 JPEG portrait photo containing one human face. No EXIF GPS.

**Steps:**
1. User sends `POST /openai/v1/chat/completions` with base64 image to `http://localhost:8003/openai/v1/chat/completions`
2. ProxyInterceptor parses `messages[0].content[1].image_url.url` → extracts base64 PNG
3. `AdversarialProtector.protect(image, session_id)` with ε=4/255 → protected image
4. `SemanticVerifier.verify(original, protected)` → SSIM ≥ 0.95, PSNR ≥ 35 dB, CLIP cosine ≥ 0.92
5. Request forwarded to `https://api.openai.com/v1/chat/completions` with protected image
6. GPT-4o responds with text: "I see a person in the image with..."
7. Proxy returns response unchanged (no image in response → no reversal needed)

**Expected output:** 
- GPT-4o response returned correctly to user
- `X-Privacy-Agent: 1` header in forwarded request
- Session logged in SQLite with `ssim ≥ 0.95`, `clip_cosine ≥ 0.92`, `psnr_db ≥ 35`
- User never knows protection was applied

**Pass criteria:** SSIM ≥ 0.95, CLIP ≥ 0.92, PSNR ≥ 35 dB, total overhead ≤ 200ms

---

## Scenario 2: DALL-E Image Editing with Reversal

**Setup:** User sends photo for DALL-E inpainting via proxy.

**Input:** 1024×1024 PNG image with inpainting mask; prompt: "replace the background with a beach"

**Steps:**
1. `POST /openai/v1/images/edits` with multipart form (image + mask + prompt)
2. ProxyInterceptor extracts image from multipart, protects with ε=4/255
3. Forward modified request to OpenAI images edits endpoint
4. OpenAI returns JSON: `{"data": [{"b64_json": "..."}]}`
5. ProxyInterceptor detects `b64_json` in response, applies reversal using stored session mask
6. Returns response with noise-reversed image to user

**Expected output:**
- Returned image is the DALL-E edited version with noise removed from non-edited regions
- Recovery SSIM ≥ 0.99 on background regions that DALL-E preserved
- `reversal_applied=1` in session record

**Pass criteria:** b64_json in response correctly reversed; user receives clean edited image

---

## Scenario 3: Quality Gate Rejection — Epsilon Too High

**Setup:** Admin misconfigures epsilon=30 (out of safe range) in agent_config.yaml.

**Steps:**
1. ProxyInterceptor receives request with personal photo
2. `AdversarialProtector.protect(image, session_id)` with ε=30/255
3. `SemanticVerifier.verify()` → SSIM = 0.82 < 0.95 gate fails
4. Agent rejects forwarding the request

**Expected output:**
- HTTP 422 response: `{"error": "protection_failed", "detail": "SSIM 0.8200 < threshold 0.95 — reduce epsilon"}`
- Original image never sent to API
- Error count incremented in metrics

**Pass criteria:** Request blocked; no API call made; 422 returned with clear error message

---

## Scenario 4: Privacy Threat Analysis on Document Photo

**Setup:** User uploads photo of their ID card to analyze privacy risk before editing.

**Command:** `python agent/main.py analyze id_photo.jpg --context "I want to blur the ID number"`

**Input:** Photo of national ID card — contains face, text (name, ID number, DOB), no GPS EXIF.

**Steps:**
1. ThreatAnalyzer.analyze() runs locally (no external calls for analysis)
2. Face detection: 1 face detected
3. Text heuristic: high edge density → has_text=True
4. EXIF check: no GPS found
5. Risk scoring: face=+2, text=+1, no GPS=0 → risk_score=3 → risk_level="high"
6. If LLM available: Claude generates threat report JSON with attack scenarios
7. Recommend epsilon=4, mode="pixel", strip EXIF

**Expected output:**
```
Risk Level:       HIGH
PII Detected:     face, visible_text
Recommended ε:    4/255
Confidence:       85%

Attack Scenarios:
  1. API provider logs raw request — ID card stored in plaintext
  2. Model inversion attack — face + text recoverable from training data
  3. Network interception — TLS-terminating corporate proxy reads cleartext

Recommendations:
  • Use epsilon=4/255 noise protection for this risk level
  • Prefer Ollama offline mode to avoid cloud data exposure
  • Strip EXIF metadata before any upload
```

**Pass criteria:** risk_level=high or critical; face and text in pii_types; epsilon=4 recommended

---

## Scenario 5: Anthropic Claude Vision Route

**Setup:** User uses Claude API for image description via proxy.

**Input:** POST to `/anthropic/v1/messages` with base64 image in Anthropic message format.

**Request body structure:**
```json
{
  "model": "claude-opus-4-8",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
      {"type": "text", "text": "Describe this image"}
    ]
  }]
}
```

**Steps:**
1. ProxyInterceptor receives `POST /anthropic/v1/messages`
2. Walks JSON structure; finds `source.data` base64 image in content array
3. Applies protection; re-encodes to PNG
4. Forwards to `https://api.anthropic.com/v1/messages` with all headers preserved
5. Claude returns text response (no image in response)
6. Proxy returns Claude response unchanged

**Expected output:**
- Claude processes the semantically-equivalent protected image and returns description
- All original headers (anthropic-version, x-api-key) passed through unchanged
- SSIM ≥ 0.95, CLIP ≥ 0.92

**Pass criteria:** Anthropic headers preserved; base64 in `source.data` correctly extracted and replaced

---

## Scenario 6: Graceful Degradation — All LLM Providers Unavailable

**Setup:** No ANTHROPIC_API_KEY, no OPENAI_API_KEY set. Ollama not running.

**Input:** POST to `/openai/v1/chat/completions` with image.

**Steps:**
1. Image protection proceeds normally (LLM not required for protect/reverse)
2. Request forwarded to real OpenAI API
3. If threat analysis is separately requested, LLMClient tries all 3 providers and gets errors
4. ThreatAnalyzer falls back to heuristic scoring

**Expected output:**
- Proxy still works for image protection (no LLM needed for protect/reverse cycle)
- Threat analysis returns heuristic report: `[LLM unavailable — using heuristic analysis]`
- SSIM/PSNR/CLIP gates still enforced
- Error message visible only in threat report; proxy operation unaffected

**Pass criteria:** Proxy protects images without LLM; threat analysis gracefully degrades to heuristics

---

## Scenario 7: Weekly Knowledge Crawler Deduplication

**Setup:** Run `update-knowledge` twice in a row.

**Steps:**
1. First run: crawls ArXiv cs.CR + cs.CV → finds 15 papers → adds 15 to SECOND-KNOWLEDGE-BRAIN.md → hashes saved to `knowledge_hashes` table
2. Second run (immediately after): crawls same sources → finds same 15 papers → all have matching hashes → 0 papers added

**Expected output:**
```
[KnowledgeUpdater] Added 0 new entries. Next run: weekly Sunday 02:00
```
SECOND-KNOWLEDGE-BRAIN.md has no duplicate rows.

**Pass criteria:** Second run adds 0 papers; no duplicates in SECOND-KNOWLEDGE-BRAIN.md

---

## Scenario 8: Full REST API Integration Test

**Setup:** Proxy running at localhost:8003. Test client calls all endpoints.

**Steps:**
1. `GET /health` → `{"status": "ok"}`
2. `GET /metrics` → Prometheus text format with `privacy_proxy_sessions_total`
3. `POST /openai/v1/chat/completions` with image → protected request forwarded → `X-Privacy-Agent: 1`
4. `POST /anthropic/v1/messages` with image → protected request forwarded
5. `POST /openai/v1/images/edits` multipart → image protected → response image reversed
6. `POST /ollama/api/generate` → passed through without protection (local model, no protection needed)

**Expected output:**
- All 4 API format types handled correctly
- Ollama route has `X-Privacy-Agent: no-image` header (no modification)
- OpenAI and Anthropic routes have `X-Privacy-Agent: 1`
- SSIM ≥ 0.95 for all protected images

**Pass criteria:** All 6 endpoints respond correctly; protection applied to cloud routes only
