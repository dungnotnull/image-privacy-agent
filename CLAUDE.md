# image-privacy-agent — Image Privacy Protection & Adversarial Defense Agent

**Tagline:** A local privacy proxy that protects your personal images from interception during LLM API calls — imperceptible to the AI, invisible to hackers.

**Build Phase:** Phase 0 — Architecture complete, development ready.

---

## Problem Statement

When users upload personal photos to LLM services (ChatGPT, Claude, Gemini) for analysis or editing, the raw pixel data travels over the internet and is stored in API logs, model training queues, and provider infrastructure. A network-layer attacker or malicious insider can intercept and reconstruct these images. This agent runs locally as a transparent HTTP proxy: it applies a reversible, session-keyed adversarial noise mask to every outgoing image before it reaches the LLM API, and reverses the mask on the returned edited result. The AI still understands the semantic content; attackers only see unintelligible noise.

---

## Agent Architecture

```
User App (ChatGPT client / API caller)
        │ POST /openai/* or /anthropic/*
        ▼
┌──────────────────────────────────────────┐
│  PrivacyProxyOrchestrator                │
│  ┌────────────────────────────────────┐  │
│  │ 1. ProxyInterceptor                │  │  ← FastAPI proxy, port 8003
│  │    parse image from request        │  │
│  ├────────────────────────────────────┤  │
│  │ 2. AdversarialProtector            │  │  ← FGSM noise + DCT masking
│  │    protect(image, session_key)     │  │
│  ├────────────────────────────────────┤  │
│  │ 3. SemanticVerifier                │  │  ← CLIP cosine + SSIM gate
│  │    verify SSIM ≥ 0.95 (skip send   │  │
│  │    if gate fails — image too noisy)│  │
│  ├────────────────────────────────────┤  │
│  │ 4. Forward to Real API             │  │  ← httpx async to OpenAI/Anthropic
│  ├────────────────────────────────────┤  │
│  │ 5. Reverse protection on response  │  │  ← subtract stored noise mask
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │ ThreatAnalyzer                     │  │  ← on-demand LLM threat report
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
        │
        ▼
LLM API (OpenAI / Anthropic / Ollama)
        │
        ▼
MemoryManager (SQLite — session store)
```

---

## Module List (`agent/modules/`)

| File | Responsibility |
|------|---------------|
| `adversarial_protector.py` | Session-keyed pixel-space noise (L∞ ≤ 4/255) + DCT block masking; `protect()` and `reverse()` |
| `proxy_interceptor.py` | FastAPI HTTP proxy: parse images from multipart/JSON, apply protection, forward, reverse response |
| `semantic_verifier.py` | CLIP cosine similarity + SSIM/PSNR quality gates; block sending if semantic drift > threshold |
| `threat_analyzer.py` | LLM-based image privacy threat modeling: detect PII/faces/location; recommend protection level |

---

## HuggingFace Models

| Model ID | Task | Why chosen |
|----------|------|------------|
| `openai/clip-vit-large-patch14` | Semantic preservation check (cosine similarity before/after noise) | CLIP-L achieves 75.5% zero-shot ImageNet; industry standard for semantic alignment |
| `google/vit-base-patch16-224` | Image quality assessment & perceptual scoring | ViT-B/16 ImageNet 81.8% accuracy; fast inference for quality gating |
| `BAAI/bge-large-en-v1.5` | Text embedding for threat knowledge retrieval from SECOND-KNOWLEDGE-BRAIN.md | MTEB leaderboard #1 dense retrieval; reused from other cluster agents |

---

## LLM API Integration

| Provider | Role in this agent |
|----------|-------------------|
| Claude (primary) | Privacy threat modeling; classify PII types visible in image; generate security recommendations; synthesize knowledge brain papers into actionable advice |
| OpenAI GPT-4o (fallback) | Same as above; also used as the target API being proxied (so we test our own proxy against it) |
| Ollama llama3 (offline) | Full offline mode — threat analysis without any external API call, for maximum privacy |

---

## Knowledge Crawl Sources

| Source | Categories/Queries | Frequency |
|--------|-------------------|-----------|
| ArXiv XML API | `cs.CR` (cryptography/security), `cs.CV` (computer vision adversarial) | Weekly (Sunday 02:00) |
| Semantic Scholar Graph API | "adversarial examples privacy", "image steganography", "differential privacy vision" | Weekly |
| IEEE S&P proceedings | adversarial ML, image privacy | Weekly |
| USENIX Security proceedings | ML privacy attacks, membership inference, model inversion | Weekly |
| Papers with Code | adversarial robustness leaderboard, image privacy | Weekly |

---

## Supporting Tools (`tools/`)

| File | Description |
|------|-------------|
| `knowledge_updater.py` | Crawl ArXiv cs.CR+cs.CV + Semantic Scholar → append to SECOND-KNOWLEDGE-BRAIN.md weekly |
| `llm_client.py` | Unified Claude/OpenAI/Ollama client with exponential backoff and cost tracking |
| `hf_model_manager.py` | Lazy-load CLIP/ViT models with CUDA auto-detection and 600s idle unload |

---

## Active Development Tasks

- [ ] Phase 0: Architecture & research complete (this file)
- [ ] Phase 1: `adversarial_protector.py` — pixel-space + DCT noise engine
- [ ] Phase 2: `proxy_interceptor.py` — HTTP proxy middleware (OpenAI + Anthropic formats)
- [ ] Phase 3: `semantic_verifier.py` — CLIP + SSIM quality gates
- [ ] Phase 4: `threat_analyzer.py` — LLM threat modeling
- [ ] Phase 5: `orchestrator.py` — wire all modules into decision loop
- [ ] Phase 6: `SECOND-KNOWLEDGE-BRAIN.md` first crawl run
- [ ] Phase 7: Docker packaging + integration tests

---

## Quality Gates (Non-Negotiable)

1. **SSIM ≥ 0.95** on protected vs original (noise is imperceptible to humans)
2. **CLIP cosine ≥ 0.92** between original and protected (LLM semantic understanding preserved)
3. **Recovery SSIM ≥ 0.99** on unedited regions after reversal
4. **Zero key reuse**: every session uses a unique 256-bit secret derived from UUID + system entropy
5. **No local plaintext storage** of original images — only the noise mask is stored (not the image itself)
