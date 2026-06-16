<h1 align="center">Image Privacy Agent</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-ready-blue?logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
  <img src="https://img.shields.io/badge/Open--Source-ready-success" alt="Open Source Ready"/>
</p>

<p align="center">
  <b>A local privacy proxy that protects your personal images during LLM API calls.</b><br/>
  Imperceptible to humans. Invisible to hackers. Reversible on return.
</p>

---

## What is Image Privacy Agent?

**Image Privacy Agent** runs as a **localhost HTTP reverse proxy**. It intercepts every outgoing image byte, applies a **reversible, session-keyed adversarial noise mask** before it reaches the LLM API, and reverses the mask on the returned edited result. The AI still understands the semantic content; attackers only see unintelligible noise.

> **No raw pixel data ever leaves your machine.**

---

## Key Features

| Feature | Description |
|---------|-------------|
| Reversible Noise Protection | Session-keyed L-infinity pixel-space noise + optional DCT block masking |
| Semantic Quality Gate | CLIP + SSIM verification ensures the LLM still sees your image correctly |
| Lossless Reversal | Protected images are reversed on the response path; unedited regions recover >=99% SSIM |
| Multi-Provider Proxy | Supports OpenAI (ChatGPT, DALL-E), Anthropic (Claude), and Ollama (local) |
| Privacy Threat Analysis | Local PII detection (faces, text, GPS EXIF) + LLM-powered risk reports |
| Prometheus Metrics | Sessions, errors, latency histograms, average SSIM/CLIP scores |
| Knowledge Crawler | Auto-crawls ArXiv cs.CR + cs.CV + Semantic Scholar for latest adversarial privacy research |
| Docker Ready | Single-command deployment with Ollama sidecar for fully offline mode |

---

## Architecture

```
User App (ChatGPT client / API caller)
        |
        |  POST /openai/* or /anthropic/*
        v
+-------------------------------------------------------------+
|  PrivacyProxyOrchestrator                                    |
|  +-------------------------------------------------------+  |
|  | 1. ProxyInterceptor                                   |  |
|  |    parse image from request (multipart / base64)      |  |
|  +-------------------------------------------------------+  |
|  | 2. AdversarialProtector                               |  |
|  |    protect(image, session_key) -> keyed noise        |  |
|  |    L-infinity <= 4/255 + optional DCT masking         |  |
|  +-------------------------------------------------------+  |
|  | 3. SemanticVerifier                                   |  |
|  |    SSIM >= 0.95 gate                                  |  |
|  |    CLIP cosine >= 0.92 gate                           |  |
|  |    BLOCK if gate fails -> 422 error                   |  |
|  +-------------------------------------------------------+  |
|  | 4. Forward to Real API (httpx async)                  |  |
|  +-------------------------------------------------------+  |
|  | 5. Reverse protection on response                     |  |
|  |    subtract stored noise mask -> clean image          |  |
|  +-------------------------------------------------------+  |
|  +-------------------------------------------------------+  |
|  | ThreatAnalyzer (on-demand)                            |  |
|  |    Face detection . EXIF GPS scan . Text heuristic     |  |
|  |    LLM risk report (image NEVER sent to LLM)          |  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
        |
        v
LLM API (OpenAI / Anthropic / Ollama)
        |
        v
MemoryManager (SQLite - session store, cost log, paper hashes)
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- Docker (optional, for containerized deployment)
- Ollama (optional, for offline/privacy mode)

### Installation

```
# Clone the repository
git clone https://github.com/dungnotnull/image-privacy-agent.git
cd image-privacy-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```
# Copy environment template
cp config/.env.example .env

# Edit .env with your API keys (optional - proxy works without them)
nano .env
```

Minimum .env for threat analysis:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
PRIVACY_MASTER_SECRET=
```

Maximum privacy (offline only):
```
PRIVACY_MODE=true
OLLAMA_BASE_URL=http://localhost:11434
```

### Start the Proxy

```
# Development mode
python agent/main.py serve --port 8003 --epsilon 4 --mode pixel

# Production with Docker
docker-compose up -d
```

### Configure Your LLM Client

Point your application at the local proxy instead of the real API:

```
export OPENAI_BASE_URL=http://localhost:8003/openai
export ANTHROPIC_BASE_URL=http://localhost:8003/anthropic
```

That is it. Every image you send is now automatically protected.

---

## CLI Commands

```
# Start proxy server
python agent/main.py serve --host 127.0.0.1 --port 8003 --epsilon 4

# Analyze privacy risk of a local image
python agent/main.py analyze photo.jpg --context "LinkedIn headshot"

# Benchmark protection quality
python agent/main.py benchmark photo.jpg

# Trigger manual knowledge crawl
python agent/main.py update-knowledge

# Show LLM API cost breakdown
python agent/main.py cost-report

# Show proxy runtime status
python agent/main.py status
```

---

## Quality Gates

Every protected image must pass **three independent gates** before leaving your machine:

| Gate | Threshold | What It Means |
|------|-----------|---------------|
| SSIM | >= 0.95 | Human-eye imperceptibility |
| PSNR | >= 35 dB | Well above perceptual noise floor |
| CLIP Cosine | >= 0.92 | LLM semantic understanding preserved |

If any gate fails, the request is **rejected with HTTP 422** and the original image is **never sent**.

---

## Running Tests

```
# Run all tests
pytest tests/test_agent.py -v

# Run specific module tests
pytest tests/test_agent.py::TestAdversarialProtector -v
pytest tests/test_agent.py::TestSemanticVerifier -v
pytest tests/test_agent.py::TestProxyIntegration -v
```

---

## Docker Deployment

```
# CPU mode
docker-compose up -d privacy-proxy

# With GPU (CUDA)
docker-compose --profile gpu up -d privacy-proxy-gpu

# With Ollama sidecar (fully offline)
docker-compose --profile ollama up -d
```

Health checks are built in:
- Proxy: curl http://localhost:8003/health
- Ollama: curl http://localhost:11434/api/tags

---

## Prometheus Metrics

The proxy exposes production-ready metrics at GET /metrics:

```
# HELP privacy_proxy_sessions_total Total proxy sessions
# TYPE privacy_proxy_sessions_total counter
privacy_proxy_sessions_total 1427

# HELP privacy_proxy_errors_total Total proxy errors
# TYPE privacy_proxy_errors_total counter
privacy_proxy_errors_total 3

# HELP privacy_proxy_avg_latency_ms Average proxy overhead latency
# TYPE privacy_proxy_avg_latency_ms gauge
privacy_proxy_avg_latency_ms 47.32
```

---

## How It Works

### Protection Pipeline

1. **Parse** - Extract images from JSON (data:image/... URIs) or multipart form data
2. **Key Derivation** - PBKDF2(session_id + MASTER_SECRET, salt, 100_000, dklen=32)
3. **Noise Generation** - PCG64 PRNG seeded from key -> uniform L-infinity noise (+/-4 gray levels)
4. **DCT Masking** (optional) - Zero high-frequency coefficients in 8x8 blocks
5. **Quality Gate** - SSIM + PSNR + CLIP cosine verification
6. **Forward** - httpx.AsyncClient sends protected PNG to real API
7. **Response Scan** - Detect returned images in JSON body
8. **Reversal** - Subtract stored noise mask, restore DCT coefficients
9. **Return** - Clean image returned to caller

### Cryptographic Guarantees

- **256-bit session keys** via PBKDF2-HMAC-SHA256
- **Unique keys per session** - UUID4 namespace makes collision probability 1/2^122
- **No key material leaves the host** - master secret stored in ~/.config/ with 0600 permissions
- **No plaintext image storage** - only noise masks are cached in memory; images are never persisted

---

## Project Structure

```
image-privacy-agent/
agent/
  __init__.py
  main.py                    # Click CLI entry point
  orchestrator.py            # PrivacyProxyOrchestrator (wires all modules)
  memory/
    __init__.py
    memory_manager.py        # SQLite WAL session store
  modules/
    __init__.py
    adversarial_protector.py  # Noise engine + DCT masking
    proxy_interceptor.py      # FastAPI HTTP proxy
    semantic_verifier.py      # CLIP + SSIM quality gate
    threat_analyzer.py        # PII detection + LLM risk report
tools/
  __init__.py
  knowledge_updater.py       # ArXiv + Semantic Scholar crawler
  llm_client.py              # Claude -> OpenAI -> Ollama fallback
  hf_model_manager.py        # CLIP/ViT/BGE/BLIP lazy loader
config/
  .env.example               # Environment variable template
  agent_config.yaml          # Runtime configuration
  config_loader.py           # YAML + env overlay
docker/
  Dockerfile                 # python:3.12-slim, non-root user
  docker-compose.yml         # Proxy + Ollama sidecar
tests/
  test-scenarios.md          # 8 end-to-end scenarios
  test_agent.py              # 41 unit and integration tests
requirements.txt             # Pinned dependencies
CLAUDE.md                    # Agent identity and architecture
PROJECT-detail.md            # Full technical specification
PROJECT-DEVELOPMENT-PHASE-TRACKING.md
SECOND-KNOWLEDGE-BRAIN.md    # Self-updating research knowledge base
```

---

## Supported Models and Hardware

| Component | Default Model | Hardware |
|-----------|-------------|----------|
| Semantic Verification | openai/clip-vit-large-patch14 | CPU / CUDA auto-detect |
| Image Captioning | Salesforce/blip-image-captioning-base | CPU / CUDA auto-detect |
| Knowledge Retrieval | BAAI/bge-large-en-v1.5 | CPU / CUDA auto-detect |
| Offline LLM | llama3 via Ollama | CPU / optional GPU |

Models are **lazy-loaded on first use** and **auto-unloaded after 10 minutes** of idle time to minimize RAM footprint.

---

## Threat Model and Defense Depth

| Attack Vector | Defense |
|---------------|---------|
| API provider logs raw pixels | Protected image has no perceptual or semantic relationship to original |
| TLS MITM / corporate proxy | Interceptor sees only noised pixels; cannot reverse without session key |
| Model inversion from training data | Noised inputs break pixel-level reconstruction |
| Membership inference | Per-session noise makes each upload unique; cannot correlate |
| Frequency-domain extraction | Optional DCT masking removes high-frequency information |
| EXIF GPS leakage | Local EXIF stripping recommended + ThreatAnalyzer warns |
| Face recognition in transit | L-infinity noise disrupts pixel-level facial feature extraction |

---

## Contributing

We welcome contributions!

1. Fork the repository
2. Create your feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m "Add amazing feature")
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

### Development Setup

```bash
pip install -r requirements.txt
pytest tests/test_agent.py -v
```

---

## License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## Acknowledgments

- Goodfellow et al. - FGSM adversarial examples (ICLR 2015)
- Madry et al. - PGD adversarial training (ICLR 2018)
- Shan et al. - Fawkes face cloaking (USENIX Security 2020)
- Radford et al. - CLIP semantic alignment (ICML 2021)
- Wang et al. - SSIM perceptual metric (IEEE TIP 2004)

Built with love for privacy-conscious developers, security researchers, and anyone who believes your photos should stay yours.
