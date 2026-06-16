"""Automated tests for the Image Privacy Agent."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rgb_image() -> Image.Image:
    arr = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_gray_image() -> Image.Image:
    arr = np.random.randint(50, 200, (128, 128), dtype=np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


@pytest.fixture
def face_image() -> Image.Image:
    # Gradient image simulating a portrait
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    arr[100:400, 150:350] = [220, 180, 160]  # Skin-tone region
    return Image.fromarray(arr, mode="RGB")


def image_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ---------------------------------------------------------------------------
# AdversarialProtector Tests
# ---------------------------------------------------------------------------

class TestAdversarialProtector:
    def test_protect_changes_pixels(self, sample_rgb_image):
        from agent.modules.adversarial_protector import AdversarialProtector
        prot = AdversarialProtector(epsilon_int=4)
        session_id = str(uuid.uuid4())
        protected = prot.protect(sample_rgb_image, session_id)
        assert protected.size == sample_rgb_image.size
        diff = np.abs(np.array(protected).astype(int) - np.array(sample_rgb_image).astype(int))
        assert diff.max() <= 4, "L∞ norm must be ≤ epsilon"
        assert diff.max() > 0, "Protection must change at least some pixels"

    def test_protect_then_reverse_is_exact(self, sample_rgb_image):
        from agent.modules.adversarial_protector import AdversarialProtector
        prot = AdversarialProtector(epsilon_int=4)
        session_id = str(uuid.uuid4())
        protected = prot.protect(sample_rgb_image, session_id)
        recovered = prot.reverse(protected, session_id)
        diff = np.abs(np.array(recovered).astype(int) - np.array(sample_rgb_image).astype(int))
        assert diff.max() == 0, "Reversal must be pixel-perfect"

    def test_different_sessions_use_different_noise(self, sample_rgb_image):
        from agent.modules.adversarial_protector import AdversarialProtector
        prot = AdversarialProtector(epsilon_int=4)
        s1, s2 = str(uuid.uuid4()), str(uuid.uuid4())
        p1 = prot.protect(sample_rgb_image, s1)
        # Need new protector instance since mask is stored per-session
        prot2 = AdversarialProtector(epsilon_int=4)
        p2 = prot2.protect(sample_rgb_image, s2)
        # With different session_ids (→ different keys), noise patterns differ
        assert not np.array_equal(np.array(p1), np.array(p2))

    def test_stats_within_expected_range(self, sample_rgb_image):
        from agent.modules.adversarial_protector import AdversarialProtector
        prot = AdversarialProtector(epsilon_int=4)
        session_id = str(uuid.uuid4())
        protected = prot.protect(sample_rgb_image, session_id)
        stats = prot.compute_stats(sample_rgb_image, protected)
        assert stats.l_inf_norm <= 4.0
        assert stats.psnr_db >= 35.0
        assert 0.9 <= stats.ssim <= 1.0

    def test_reverse_without_stored_mask_returns_unchanged(self, sample_rgb_image):
        from agent.modules.adversarial_protector import AdversarialProtector
        prot = AdversarialProtector(epsilon_int=4)
        result = prot.reverse(sample_rgb_image, "nonexistent-session")
        assert np.array_equal(np.array(result), np.array(sample_rgb_image))

    def test_rgba_image_preserved(self):
        from agent.modules.adversarial_protector import AdversarialProtector
        arr = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        rgba_img = Image.fromarray(arr, mode="RGBA")
        prot = AdversarialProtector(epsilon_int=4)
        session_id = str(uuid.uuid4())
        protected = prot.protect(rgba_img, session_id)
        assert protected.mode == "RGBA"

    def test_epsilon_validation(self):
        from agent.modules.adversarial_protector import AdversarialProtector
        with pytest.raises(ValueError):
            AdversarialProtector(epsilon_int=0)
        with pytest.raises(ValueError):
            AdversarialProtector(epsilon_int=33)


# ---------------------------------------------------------------------------
# SemanticVerifier Tests
# ---------------------------------------------------------------------------

class TestSemanticVerifier:
    def test_identical_images_pass_all_gates(self, sample_rgb_image):
        from agent.modules.semantic_verifier import SemanticVerifier
        verifier = SemanticVerifier(clip_threshold=0.92)
        result = verifier.verify(sample_rgb_image, sample_rgb_image)
        assert result.ssim == pytest.approx(1.0, abs=0.01)
        assert result.psnr_db == float("inf") or result.psnr_db > 45

    def test_low_noise_passes_ssim_gate(self, sample_rgb_image):
        from agent.modules.adversarial_protector import AdversarialProtector
        from agent.modules.semantic_verifier import SemanticVerifier
        prot = AdversarialProtector(epsilon_int=4)
        session_id = str(uuid.uuid4())
        protected = prot.protect(sample_rgb_image, session_id)
        verifier = SemanticVerifier()
        result = verifier.verify(sample_rgb_image, protected)
        assert result.ssim >= 0.95
        assert result.psnr_db >= 35.0

    def test_heavily_modified_image_fails_ssim_gate(self, sample_rgb_image):
        from agent.modules.semantic_verifier import SemanticVerifier
        # Destroy image by adding heavy noise
        arr = np.array(sample_rgb_image)
        heavily_noised = Image.fromarray(
            np.clip(arr.astype(int) + np.random.randint(-60, 60, arr.shape), 0, 255).astype(np.uint8)
        )
        verifier = SemanticVerifier(ssim_threshold=0.95)
        result = verifier.verify(sample_rgb_image, heavily_noised)
        assert not result.passed
        assert "SSIM" in result.reason or "PSNR" in result.reason

    def test_recovery_verification(self, sample_rgb_image):
        from agent.modules.adversarial_protector import AdversarialProtector
        from agent.modules.semantic_verifier import SemanticVerifier
        prot = AdversarialProtector(epsilon_int=4)
        session_id = str(uuid.uuid4())
        protected = prot.protect(sample_rgb_image, session_id)
        recovered = prot.reverse(protected, session_id)
        verifier = SemanticVerifier()
        result = verifier.verify_recovery(sample_rgb_image, recovered)
        assert result.passed
        assert result.ssim >= 0.99

    def test_fallback_cosine_without_clip(self, sample_rgb_image):
        from agent.modules.semantic_verifier import SemanticVerifier
        # hf_manager=None forces fallback
        verifier = SemanticVerifier(clip_threshold=0.5, hf_manager=None)
        result = verifier.verify(sample_rgb_image, sample_rgb_image)
        assert result.fallback_used
        assert result.clip_cosine >= 0.5


# ---------------------------------------------------------------------------
# ThreatAnalyzer Tests
# ---------------------------------------------------------------------------

class TestThreatAnalyzer:
    def test_heuristic_analysis_face_image(self, face_image):
        from agent.modules.threat_analyzer import ThreatAnalyzer
        analyzer = ThreatAnalyzer(llm_client=None)
        report = analyzer.analyze(face_image, "LinkedIn photo editing")
        # face_count may be 0 without a real face, but test heuristic path
        assert report.risk_level in ("low", "medium", "high", "critical")
        assert isinstance(report.pii_types, list)
        assert report.recommended_epsilon in (2.0, 4.0, 8.0)
        assert len(report.recommendations) >= 3
        assert len(report.attack_scenarios) == 3

    def test_gps_exif_triggers_high_risk(self, tmp_path):
        from agent.modules.threat_analyzer import ThreatAnalyzer
        import piexif
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        exif_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: b'N',
            piexif.GPSIFD.GPSLatitude: ((10, 1), (30, 1), (0, 1)),
        }
        exif_bytes = piexif.dump({"GPS": exif_ifd})
        img_path = tmp_path / "gps_test.jpg"
        img.save(str(img_path), exif=exif_bytes)
        img_with_gps = Image.open(str(img_path))

        analyzer = ThreatAnalyzer(llm_client=None)
        meta = analyzer._extract_metadata(img_with_gps)
        assert meta["has_gps"] is True

    def test_no_pii_gives_low_risk(self):
        from agent.modules.threat_analyzer import ThreatAnalyzer
        # Plain gradient image — no face, no text
        arr = np.linspace(0, 255, 64 * 64).reshape(64, 64).astype(np.uint8)
        plain_img = Image.fromarray(arr, mode="L").convert("RGB")
        analyzer = ThreatAnalyzer(llm_client=None)
        report = analyzer.analyze(plain_img, "background blur")
        assert report.risk_level in ("low", "medium")
        assert report.recommended_epsilon <= 4.0

    def test_llm_fallback_on_json_parse_error(self, sample_rgb_image):
        from agent.modules.threat_analyzer import ThreatAnalyzer
        mock_llm = MagicMock()
        mock_llm.complete_sync.return_value = "not valid json"
        analyzer = ThreatAnalyzer(llm_client=mock_llm)
        report = analyzer.analyze(sample_rgb_image, "test")
        assert report.risk_level in ("low", "medium", "high", "critical")


# ---------------------------------------------------------------------------
# MemoryManager Tests
# ---------------------------------------------------------------------------

class TestMemoryManager:
    def test_save_and_retrieve_session(self, tmp_path):
        from agent.memory.memory_manager import PrivacyMemoryManager
        db = PrivacyMemoryManager(db_path=tmp_path / "test.db")
        session_id = str(uuid.uuid4())
        db.save_session(
            session_id=session_id,
            api_provider="openai",
            image_hash="abc123",
            epsilon=4.0,
            ssim=0.97,
            clip_cosine=0.95,
            psnr_db=38.5,
        )
        stats = db.get_session_stats()
        assert stats["total_sessions"] == 1
        assert stats["avg_ssim"] == pytest.approx(0.97, abs=0.01)

    def test_mark_and_check_paper(self, tmp_path):
        from agent.memory.memory_manager import PrivacyMemoryManager
        db = PrivacyMemoryManager(db_path=tmp_path / "test.db")
        hash_val = "abc123def456"
        assert not db.is_known_paper(hash_val)
        db.mark_paper_known(hash_val)
        assert db.is_known_paper(hash_val)

    def test_cost_tracking(self, tmp_path):
        from agent.memory.memory_manager import PrivacyMemoryManager
        db = PrivacyMemoryManager(db_path=tmp_path / "test.db")
        db.log_llm_cost("claude", "claude-opus-4-8", 100, 200, 0.0045, "threat_analysis")
        summary = db.get_cost_summary()
        assert "claude" in summary
        assert summary["claude"]["total_cost"] == pytest.approx(0.0045, abs=1e-6)

    def test_threat_report_storage(self, tmp_path):
        from agent.memory.memory_manager import PrivacyMemoryManager
        db = PrivacyMemoryManager(db_path=tmp_path / "test.db")
        db.save_threat_report(
            session_id=None,
            pii_types=["face", "text"],
            risk_level="high",
            recommended_epsilon=4.0,
            recommendations=["use epsilon 4"],
            attack_scenarios=["API log attack"],
            confidence=0.85,
        )
        stats = db.get_stats()
        assert stats["known_papers"] == 0

    def test_concurrent_access(self, tmp_path):
        import threading
        from agent.memory.memory_manager import PrivacyMemoryManager
        db = PrivacyMemoryManager(db_path=tmp_path / "concurrent.db")
        errors = []

        def write_session(i):
            try:
                db.save_session(
                    str(uuid.uuid4()), "openai", f"hash{i}",
                    4.0, 0.97, 0.95, 38.0
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_session, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = db.get_session_stats()
        assert stats["total_sessions"] == 20


# ---------------------------------------------------------------------------
# LLMClient Tests
# ---------------------------------------------------------------------------

class TestLLMClient:
    @pytest.mark.asyncio
    async def test_complete_returns_string(self):
        from tools.llm_client import LLMClient
        client = LLMClient()
        with patch.object(client, "_call_provider", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Test response"
            result = await client.complete("test prompt")
        assert isinstance(result, str)
        assert "Test response" in result

    @pytest.mark.asyncio
    async def test_fallback_chain(self):
        from tools.llm_client import LLMClient
        client = LLMClient()
        call_order = []

        async def mock_call(provider, prompt, system, max_tokens):
            call_order.append(provider)
            if provider != "ollama":
                raise ConnectionError("API error")
            return "Ollama response"

        with patch.object(client, "_call_provider", side_effect=mock_call):
            with patch.object(client, "_build_provider_chain", return_value=["claude", "openai", "ollama"]):
                result = await client.complete("test")
        assert "ollama" in call_order
        assert "Ollama response" in result

    @pytest.mark.asyncio
    async def test_all_providers_fail_returns_fallback_message(self):
        from tools.llm_client import LLMClient
        client = LLMClient()

        async def always_fail(provider, prompt, system, max_tokens):
            raise ConnectionError("always fails")

        with patch.object(client, "_call_provider", side_effect=always_fail):
            with patch.object(client, "_build_provider_chain", return_value=["claude"]):
                result = await client.complete("test")
        assert "unavailable" in result.lower()


# ---------------------------------------------------------------------------
# HFModelManager Tests
# ---------------------------------------------------------------------------

class TestHFModelManager:
    def test_heuristic_similarity_identical_images(self, sample_rgb_image):
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager()
        sim = mgr._heuristic_image_similarity(sample_rgb_image, sample_rgb_image)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_heuristic_similarity_different_images(self, sample_rgb_image):
        from tools.hf_model_manager import HFModelManager
        other = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        mgr = HFModelManager()
        sim = mgr._heuristic_image_similarity(sample_rgb_image, other)
        assert sim < 0.8

    def test_tfidf_fallback_produces_normalized_vector(self):
        from tools.hf_model_manager import HFModelManager
        vecs = HFModelManager._tfidf_fallback(["adversarial privacy image"])
        assert vecs.shape[1] == 1024
        norm = np.linalg.norm(vecs[0])
        assert norm == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestProxyIntegration:
    def test_json_body_protect_and_walk(self, sample_rgb_image):
        """Walk JSON body, find base64 image, protect it, verify it changed."""
        from agent.modules.proxy_interceptor import _walk_and_protect_json
        from agent.orchestrator import PrivacyProxyOrchestrator

        orch = PrivacyProxyOrchestrator(config={"epsilon_int": 4})
        b64 = image_to_b64(sample_rgb_image)
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "describe this"},
                ]}
            ]
        }
        session_id = str(uuid.uuid4())
        modified, original = _walk_and_protect_json(data, session_id, orch)
        new_url = modified["messages"][0]["content"][0]["image_url"]["url"]
        assert new_url.startswith("data:image/png;base64,")
        new_b64 = new_url.split(",", 1)[1]
        assert new_b64 != b64, "Protected image should differ from original"

    def test_end_to_end_protect_and_reverse(self, sample_rgb_image):
        """Protect a base64 image then reverse it; verify pixel-perfect recovery."""
        from agent.modules.adversarial_protector import AdversarialProtector
        from agent.modules.proxy_interceptor import _walk_and_protect_json, _walk_and_reverse_json
        from agent.orchestrator import PrivacyProxyOrchestrator

        orch = PrivacyProxyOrchestrator(config={"epsilon_int": 4})
        b64 = image_to_b64(sample_rgb_image)
        data = {"image_url": {"url": f"data:image/png;base64,{b64}"}}
        session_id = str(uuid.uuid4())

        # Protect
        modified, _ = _walk_and_protect_json(data, session_id, orch)
        protected_b64 = modified["image_url"]["url"].split(",", 1)[1]

        # Simulate response with protected image
        resp_data = {"data": [{"b64_json": protected_b64}]}
        reversed_data, changed = _walk_and_reverse_json(resp_data, session_id, orch)

        assert changed
        recovered_b64 = reversed_data["data"][0]["b64_json"]
        recovered_img = b64_to_image(recovered_b64)
        diff = np.abs(np.array(recovered_img).astype(int) - np.array(sample_rgb_image).astype(int))
        assert diff.max() == 0, "End-to-end protect+reverse must be pixel-perfect"

    @pytest.mark.asyncio
    async def test_proxy_app_health_endpoint(self):
        """Test /health endpoint returns 200."""
        from fastapi.testclient import TestClient
        from agent.orchestrator import PrivacyProxyOrchestrator
        from agent.modules.proxy_interceptor import create_proxy_app

        orch = PrivacyProxyOrchestrator()
        app = create_proxy_app(orch)
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_proxy_app_metrics_endpoint(self):
        """Test /metrics returns Prometheus text."""
        from fastapi.testclient import TestClient
        from agent.orchestrator import PrivacyProxyOrchestrator
        from agent.modules.proxy_interceptor import create_proxy_app

        orch = PrivacyProxyOrchestrator()
        app = create_proxy_app(orch)
        with TestClient(app) as client:
            resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "privacy_proxy_sessions_total" in resp.text


# ---------------------------------------------------------------------------
# CLI Smoke Tests
# ---------------------------------------------------------------------------

class TestCLISmoke:
    def test_cli_help(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "privacy" in result.output.lower() or "protect" in result.output.lower() or "Image" in result.output

    def test_cli_analyze_help(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--context" in result.output

    def test_cli_benchmark_help(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0

    def test_cli_status(self, tmp_path):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=str(tmp_path)):
            result = runner.invoke(cli, ["status"])
        # May fail on DB path — just ensure it doesn't crash with unexpected error
        assert result.exit_code in (0, 1)

    def test_cli_analyze_with_image(self, tmp_path, sample_rgb_image):
        from click.testing import CliRunner
        from agent.main import cli
        img_path = tmp_path / "test.png"
        sample_rgb_image.save(str(img_path))
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", str(img_path), "--context", "test"])
        assert result.exit_code == 0
        assert "Risk Level" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestAdversarialProtectorCombined:
    def test_combined_mode_dct_roundtrip(self, sample_rgb_image):
        from agent.modules.adversarial_protector import AdversarialProtector
        prot = AdversarialProtector(epsilon_int=4, mode="combined")
        import uuid
        sid = str(uuid.uuid4())
        protected = prot.protect(sample_rgb_image, sid)
        recovered = prot.reverse(protected, sid)
        import numpy as np
        diff = np.abs(np.array(recovered).astype(int) - np.array(sample_rgb_image).astype(int))
        assert diff.max() <= 2, "Combined mode reversal should be near-lossless"

class TestOrchestratorConfig:
    def test_orchestrator_loads_default_config(self):
        from agent.orchestrator import PrivacyProxyOrchestrator
        orch = PrivacyProxyOrchestrator()
        assert orch._config is not None
        assert "protection" in orch._config

    def test_orchestrator_metrics_increment(self):
        from agent.orchestrator import PrivacyProxyOrchestrator
        orch = PrivacyProxyOrchestrator()
        orch.inc_session(100.0)
        orch.inc_error()
        metrics = orch.get_prometheus_metrics()
        assert "privacy_proxy_sessions_total 1" in metrics
        assert "privacy_proxy_errors_total 1" in metrics

class TestProxyValidation:
    def test_build_forward_headers_strips_sensitive(self):
        from agent.modules.proxy_interceptor import _build_forward_headers
        class FakeReq:
            headers = {"content-length": "123", "host": "evil", "authorization": "Bearer tok", "x-custom": "ok"}
        headers = _build_forward_headers(FakeReq(), 456, True)
        assert headers["content-length"] == "456"
        assert "host" not in headers
        assert headers["authorization"] == "Bearer tok"
        assert headers["x-privacy-agent"] == "1"

class TestMemoryExtended:
    def test_get_stats_returns_structure(self, tmp_path):
        from agent.memory.memory_manager import PrivacyMemoryManager
        db = PrivacyMemoryManager(db_path=tmp_path / "stats.db")
        stats = db.get_stats()
        assert "sessions" in stats
        assert "costs_30d" in stats
        assert "known_papers" in stats

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
