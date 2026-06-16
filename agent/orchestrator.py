"""Core agent decision loop for the Image Privacy Agent."""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image


class PrivacyProxyOrchestrator:
    """Wire all modules together. Lazy-init for fast startup.

    Thread-safe metric counters; config auto-loaded from YAML/env.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            from config.config_loader import load_config
            config = load_config()
        self._config = config
        self._memory_inst = None
        self._llm_inst = None
        self._hf_inst = None
        self._protector_inst = None
        self._verifier_inst = None
        self._analyzer_inst = None
        self._knowledge_updater_inst = None

        self._total_sessions: int = 0
        self._error_count: int = 0
        self._total_latency_ms: float = 0.0
        self._latency_histogram: dict[str, int] = {
            "le_50": 0, "le_100": 0, "le_200": 0, "le_500": 0, "le_inf": 0,
        }
        self._start_time: float = time.time()
        self._metrics_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lazy module accessors
    # ------------------------------------------------------------------

    @property
    def memory(self):
        if self._memory_inst is None:
            from agent.memory.memory_manager import PrivacyMemoryManager
            db_path = self._config.get("memory", {}).get("db_path", "./data/privacy_agent.db")
            self._memory_inst = PrivacyMemoryManager(db_path=Path(db_path))
        return self._memory_inst

    @property
    def llm(self):
        if self._llm_inst is None:
            from tools.llm_client import LLMClient
            self._llm_inst = LLMClient(memory=self.memory)
        return self._llm_inst

    @property
    def hf(self):
        if self._hf_inst is None:
            from tools.hf_model_manager import HFModelManager
            self._hf_inst = HFModelManager.get_instance()
        return self._hf_inst

    @property
    def protector(self):
        if self._protector_inst is None:
            from agent.modules.adversarial_protector import AdversarialProtector
            epsilon = int(self._config.get("protection", {}).get("epsilon_int", os.getenv("EPSILON_INT", "4")))
            mode = self._config.get("protection", {}).get("mode", os.getenv("PROTECTION_MODE", "pixel"))
            self._protector_inst = AdversarialProtector(epsilon_int=epsilon, mode=mode)
        return self._protector_inst

    @property
    def verifier(self):
        if self._verifier_inst is None:
            from agent.modules.semantic_verifier import SemanticVerifier
            qg = self._config.get("quality_gates", {})
            self._verifier_inst = SemanticVerifier(
                ssim_threshold=float(qg.get("ssim_threshold", 0.95)),
                psnr_threshold=float(qg.get("psnr_threshold", 35.0)),
                clip_threshold=float(qg.get("clip_threshold", 0.92)),
                hf_manager=self.hf,
            )
        return self._verifier_inst

    @property
    def analyzer(self):
        if self._analyzer_inst is None:
            from agent.modules.threat_analyzer import ThreatAnalyzer
            self._analyzer_inst = ThreatAnalyzer(llm_client=self.llm, memory=self.memory, hf_manager=self.hf)
        return self._analyzer_inst

    @property
    def knowledge_updater(self):
        if self._knowledge_updater_inst is None:
            from tools.knowledge_updater import KnowledgeUpdater
            self._knowledge_updater_inst = KnowledgeUpdater(memory=self.memory)
        return self._knowledge_updater_inst

    # ------------------------------------------------------------------
    # Pipeline: protect → verify → log
    # ------------------------------------------------------------------

    def protect_image(self, image: Image.Image, session_id: Optional[str] = None) -> tuple[Image.Image, object, str]:
        """Protect an image and return (protected_image, stats, session_id)."""
        sid = session_id or str(uuid.uuid4())
        protected = self.protector.protect(image, sid)
        stats = self.protector.compute_stats(image, protected)
        return protected, stats, sid

    def verify_protection(self, original: Image.Image, protected: Image.Image) -> object:
        """Run quality gates. Returns VerificationResult."""
        return self.verifier.verify(original, protected)

    def reverse_image(self, image: Image.Image, session_id: str) -> Image.Image:
        """Reverse protection on a response image."""
        return self.protector.reverse(image, session_id)

    def record_session(
        self,
        session_id: str,
        api_provider: str,
        image_hash: str,
        stats,
        cost_usd: float = 0.0,
    ) -> None:
        """Persist session stats to SQLite memory."""
        try:
            self.memory.save_session(
                session_id=session_id,
                api_provider=api_provider,
                image_hash=image_hash,
                epsilon=getattr(stats, "l_inf_norm", 4.0),
                ssim=getattr(stats, "ssim", 0.0),
                clip_cosine=0.0,
                psnr_db=getattr(stats, "psnr_db", 0.0),
                protection_mode=getattr(stats, "protection_mode", "pixel"),
                cost_usd=cost_usd,
            )
        except Exception:
            pass

    def mark_reversal(self, session_id: str) -> None:
        try:
            self.memory.mark_reversal(session_id)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def inc_session(self, latency_ms: float) -> None:
        with self._metrics_lock:
            self._total_sessions += 1
            self._total_latency_ms += latency_ms
            if latency_ms <= 50:
                bucket = "le_50"
            elif latency_ms <= 100:
                bucket = "le_100"
            elif latency_ms <= 200:
                bucket = "le_200"
            elif latency_ms <= 500:
                bucket = "le_500"
            else:
                bucket = "le_inf"
            self._latency_histogram[bucket] += 1

    def inc_error(self) -> None:
        with self._metrics_lock:
            self._error_count += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_threat(self, image: Image.Image, context: str = "general") -> dict:
        """Run privacy threat analysis on a local image. Returns serializable dict."""
        report = self.analyzer.analyze(image, context)
        return {
            "risk_level": report.risk_level,
            "pii_types": report.pii_types,
            "recommended_epsilon": report.recommended_epsilon,
            "recommended_mode": report.recommended_mode,
            "attack_scenarios": report.attack_scenarios,
            "recommendations": report.recommendations,
            "confidence": report.confidence,
            "advisory": report.advisory_text,
            "metadata": {
                "face_count": report.face_count,
                "has_gps": report.has_gps,
                "has_text": report.has_text,
            },
        }

    def benchmark_image(self, image: Image.Image) -> dict:
        """Benchmark protection quality on a test image."""
        from agent.modules.adversarial_protector import AdversarialProtector

        results = {}
        for epsilon in [2, 4, 8]:
            prot = AdversarialProtector(epsilon_int=epsilon, mode="pixel")
            session_id = str(uuid.uuid4())
            protected = prot.protect(image, session_id)
            stats = prot.compute_stats(image, protected)
            verif = self.verifier.verify(image, protected)
            results[f"epsilon_{epsilon}"] = {
                "ssim": stats.ssim,
                "psnr_db": stats.psnr_db,
                "l_inf": stats.l_inf_norm,
                "clip_cosine": verif.clip_cosine,
                "passed_gate": verif.passed,
            }
            recovered = prot.reverse(protected, session_id)
            rev_verif = self.verifier.verify_recovery(image, recovered)
            results[f"epsilon_{epsilon}"]["recovery_ssim"] = rev_verif.ssim
        return results

    async def update_knowledge(self) -> dict:
        """Trigger manual knowledge crawl."""
        added = await self.knowledge_updater.run_weekly_update()
        return {"status": "ok", "papers_added": added}

    def get_cost_report(self) -> dict:
        return {
            "costs_30d": self.memory.get_cost_summary(days=30),
            "session_stats": self.memory.get_session_stats(),
        }

    def get_prometheus_metrics(self) -> list[str]:
        """Return Prometheus text format metrics."""
        with self._metrics_lock:
            uptime = time.time() - self._start_time
            avg_lat = self._total_latency_ms / max(self._total_sessions, 1)
            total_sessions = self._total_sessions
            error_count = self._error_count
            hist = dict(self._latency_histogram)

        lines = [
            "# HELP privacy_proxy_sessions_total Total proxy sessions",
            "# TYPE privacy_proxy_sessions_total counter",
            f"privacy_proxy_sessions_total {total_sessions}",
            "# HELP privacy_proxy_errors_total Total proxy errors",
            "# TYPE privacy_proxy_errors_total counter",
            f"privacy_proxy_errors_total {error_count}",
            "# HELP privacy_proxy_avg_latency_ms Average proxy overhead latency",
            "# TYPE privacy_proxy_avg_latency_ms gauge",
            f"privacy_proxy_avg_latency_ms {avg_lat:.2f}",
            "# HELP privacy_proxy_uptime_seconds Proxy uptime in seconds",
            "# TYPE privacy_proxy_uptime_seconds gauge",
            f"privacy_proxy_uptime_seconds {uptime:.0f}",
            "# HELP privacy_proxy_latency_bucket Latency histogram buckets",
            "# TYPE privacy_proxy_latency_bucket gauge",
        ]
        for bucket, count in hist.items():
            le = bucket.replace("le_", "")
            le_val = le if le != "inf" else "+Inf"
            lines.append(f'privacy_proxy_latency_bucket{{le="{le_val}"}} {count}')

        try:
            stats = self.memory.get_session_stats()
            lines += [
                "# HELP privacy_avg_ssim Average SSIM across all protected images",
                "# TYPE privacy_avg_ssim gauge",
                f"privacy_avg_ssim {stats.get('avg_ssim', 0):.4f}",
                "# HELP privacy_avg_clip Average CLIP cosine across all sessions",
                "# TYPE privacy_avg_clip gauge",
                f"privacy_avg_clip {stats.get('avg_clip_cosine', 0):.4f}",
            ]
        except Exception:
            pass
        return lines

    def start_scheduler(self) -> None:
        """Start weekly knowledge update scheduler."""
        try:
            self.knowledge_updater.start_scheduled()
        except Exception as exc:
            print(f"[Orchestrator] Scheduler start failed: {exc}")

    def protect_and_forward(self, image: Image.Image) -> tuple[Image.Image, object, str]:
        """Pipeline: protect -> verify -> return protected image, stats, session_id."""
        protected, stats, sid = self.protect_image(image)
        verif = self.verify_protection(image, protected)
        if not verif.passed:
            raise RuntimeError(f"Quality gate failed: {verif.reason}")
        return protected, stats, sid

    def hash_image(self, image: Image.Image) -> str:
        """Return SHA256 hex digest of raw RGBA bytes."""
        try:
            return hashlib.sha256(image.tobytes()).hexdigest()[:32]
        except Exception:
            return ""
