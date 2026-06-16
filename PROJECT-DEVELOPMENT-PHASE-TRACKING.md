# Image Privacy Agent — Development Phase Tracking

## Quantified Improvement Targets

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Protection SSIM | N/A (unprotected) | ≥ 0.95 | `skimage.metrics.structural_similarity` on 100 test images |
| CLIP semantic cosine | N/A | ≥ 0.92 | CLIP-ViT-L/14 cosine on original vs protected |
| Recovery SSIM | N/A | ≥ 0.99 | Unedited-region pixel comparison after reversal |
| Proxy overhead latency (p95) | N/A | ≤ 200ms | Measured vs direct API call; 1000-request benchmark |
| Research papers/month | 0 | ≥ 10 | COUNT entries in SECOND-KNOWLEDGE-BRAIN.md update log |

---

## Phase 0: Research & Architecture (Week 1–2)
**Goal:** Deep-dive adversarial ML and image privacy literature; finalize protection design.

### Tasks
- [x] Read idea.txt and translate user requirements
- [x] Cross-reference CLAUDE.md Cluster C improvement suggestions
- [x] Design adversarial protection strategy (pixel-space + DCT)
- [x] Design proxy architecture (FastAPI, session key management)
- [x] Write CLAUDE.md (agent identity card)
- [x] Write PROJECT-detail.md (full technical specification)
- [x] Write PROJECT-DEVELOPMENT-PHASE-TRACKING.md (this file)
- [x] Seed SECOND-KNOWLEDGE-BRAIN.md (15 papers)

**Deliverables:** CLAUDE.md, PROJECT-detail.md, this file, seeded SECOND-KNOWLEDGE-BRAIN.md

**Success criteria:** Architecture reviewed; all 7 quality gate thresholds defined and justified by literature.

**Estimated effort:** 3 person-days

---

## Phase 1: Core Modules — AdversarialProtector & SemanticVerifier (Week 3–5)
**Goal:** Implement the noise engine and quality gate.

### Tasks
- [x] `agent/modules/adversarial_protector.py`
  - [x] Pixel-space L∞ noise (PCG64 PRNG seeded from PBKDF2)
  - [x] Session mask storage in-memory dict
  - [x] `protect()` and `reverse()` methods
  - [x] DCT block masking (scipy.fft, 8×8 blocks)
  - [x] `ProtectionStats` dataclass
  - [x] PSNR helper
- [x] `agent/modules/semantic_verifier.py`
  - [x] CLIP-ViT-L/14 cosine gate
  - [x] SSIM gate (skimage)
  - [x] `VerificationResult` dataclass
  - [x] Heuristic fallback (PSNR ≥ 35 dB)

**Deliverables:** `adversarial_protector.py`, `semantic_verifier.py`

**Success criteria:** Unit tests pass; SSIM ≥ 0.95 on 100 diverse test images; CLIP cosine ≥ 0.92.

**Estimated effort:** 5 person-days

---

## Phase 2: Core Module — ProxyInterceptor (Week 5–7)
**Goal:** Implement the HTTP proxy middleware.

### Tasks
- [x] `agent/modules/proxy_interceptor.py`
  - [x] FastAPI app with OpenAI chat completions route
  - [x] FastAPI app with OpenAI images edits route (multipart)
  - [x] FastAPI app with Anthropic messages route
  - [x] Base64 image extraction from JSON body
  - [x] Multipart image file extraction
  - [x] `httpx.AsyncClient` forward logic
  - [x] Response body image detection and reversal
  - [x] `/health` and `/metrics` endpoints
  - [x] Error handling: upstream timeout, parse failure, gate failure

**Deliverables:** `proxy_interceptor.py`

**Success criteria:** Proxy overhead ≤ 200ms p95 on localhost loopback; all 3 API formats handled correctly.

**Estimated effort:** 5 person-days

---

## Phase 3: Core Module — ThreatAnalyzer (Week 7–8)
**Goal:** LLM-powered privacy threat reporting.

### Tasks
- [x] `agent/modules/threat_analyzer.py`
  - [x] EXIF metadata extraction (piexif)
  - [x] OpenCV face detection (haarcascade_frontalface_default)
  - [x] Image description construction (local ViT caption via HF model manager)
  - [x] LLM threat analysis prompt (Claude/GPT/Ollama)
  - [x] `ThreatReport` dataclass
  - [x] Risk level scoring (low/medium/high/critical)
  - [x] Protection level recommendation
  - [x] Attack scenario generation

**Deliverables:** `threat_analyzer.py`

**Success criteria:** Threat report generated in < 10s; correctly identifies faces, GPS EXIF, and visible text as high/critical risk in test cases.

**Estimated effort:** 4 person-days

---

## Phase 4: Orchestrator + Memory Manager (Week 8–9)
**Goal:** Wire all modules into the agent decision loop; implement session persistence.

### Tasks
- [x] `agent/orchestrator.py`
  - [x] `PrivacyProxyOrchestrator` class with lazy module init
  - [x] `protect_and_forward()` pipeline method
  - [x] `analyze_threat()` on-demand method
  - [x] `update_knowledge()` method
  - [x] `get_cost_report()` method
  - [x] Prometheus metrics exposition (sessions, errors, latency histogram)
  - [x] APScheduler weekly knowledge update at Sunday 02:00
- [x] `agent/memory/memory_manager.py`
  - [x] SQLite WAL mode
  - [x] `sessions` table (session_id, timestamp, image_hash, api_provider, protection_stats JSON, cost_usd)
  - [x] `threat_reports` table (session_id, pii_types JSON, risk_level, recommendations)
  - [x] `llm_cost_log` table (provider, model, prompt_tokens, completion_tokens, cost_usd)
  - [x] `knowledge_hashes` table (paper_hash, added_at)
  - [x] Threading lock for concurrent proxy requests
  - [x] `get_session_stats()`, `get_cost_summary()`

**Deliverables:** `orchestrator.py`, `memory/memory_manager.py`

**Success criteria:** 100 concurrent proxy requests handled without data race; session data persists across restarts.

**Estimated effort:** 5 person-days

---

## Phase 5: Entry Point + CLI (Week 9–10)
**Goal:** Complete agent/main.py with Click CLI and FastAPI server mount.

### Tasks
- [x] `agent/main.py`
  - [x] `serve` command: start proxy server
  - [x] `analyze` command: run threat analysis on a local image file
  - [x] `benchmark` command: run SSIM/CLIP benchmarks on test images
  - [x] `update-knowledge` command: trigger manual knowledge crawl
  - [x] `cost-report` command: show LLM API costs
  - [x] `status` command: show proxy stats (sessions, errors, uptime)

**Deliverables:** `agent/main.py`

**Success criteria:** All 6 CLI commands functional; `serve` command starts proxy in < 3s.

**Estimated effort:** 3 person-days

---

## Phase 6: Universal Tools (Week 10–12)
**Goal:** Implement knowledge_updater, llm_client, hf_model_manager.

### Tasks
- [x] `tools/knowledge_updater.py` — ArXiv cs.CR+cs.CV + Semantic Scholar crawler
- [x] `tools/llm_client.py` — Claude/OpenAI/Ollama unified client
- [x] `tools/hf_model_manager.py` — CLIP/ViT/BGE lazy loader
- [x] First manual knowledge crawl run → 15+ papers added to SECOND-KNOWLEDGE-BRAIN.md

**Deliverables:** All tools; updated SECOND-KNOWLEDGE-BRAIN.md

**Success criteria:** Knowledge updater adds ≥ 10 new papers on first run; LLM client correctly falls back from Claude → OpenAI → Ollama.

**Estimated effort:** 5 person-days

---

## Phase 7: Docker + Testing (Week 12–14)
**Goal:** Containerize, write all tests, verify end-to-end.

### Tasks
- [x] `config/agent_config.yaml` — all runtime parameters
- [x] `config/.env.example` — all required env vars documented
- [x] `docker/docker-compose.yml` — proxy agent + ollama sidecar
- [x] `docker/Dockerfile` — python:3.12-slim, non-root user, EXPOSE 8003
- [x] `tests/test-scenarios.md` — 8 complete scenarios
- [x] `tests/test_agent.py` — 40+ unit and integration tests
- [x] `requirements.txt` — all pinned dependencies

**Deliverables:** Full runnable containerized agent + test suite

**Success criteria:** All tests green; docker-compose up starts proxy in < 30s; all 7 quality gates pass on 100-image benchmark.

**Estimated effort:** 6 person-days

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| API providers change request format (break image extraction) | Medium | High | Extractor written with format-version detection; fallback to pass-through |
| CLIP model too slow for real-time gating | Medium | Medium | Skip CLIP on requests > 5 req/s (rate-limit fallback to SSIM only) |
| PNG re-encoding rejected by some API endpoints | Low | High | Detect endpoint and skip re-encoding if endpoint requires JPEG |
| User forgets to configure proxy; images sent unprotected | High | High | Startup banner warns user; optional browser extension for automatic detection |
| Session key derivation collisions | Very Low | Critical | UUID4 namespace makes collision probability 1/2^122 |

---

## Total Estimated Effort

| Phase | Person-Days |
|-------|-------------|
| Phase 0: Research & Architecture | 3 |
| Phase 1: Core Modules (protector + verifier) | 5 |
| Phase 2: ProxyInterceptor | 5 |
| Phase 3: ThreatAnalyzer | 4 |
| Phase 4: Orchestrator + Memory | 5 |
| Phase 5: Entry Point + CLI | 3 |
| Phase 6: Universal Tools | 5 |
| Phase 7: Docker + Testing | 6 |
| **Total** | **36 person-days** |
