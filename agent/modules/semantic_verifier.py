"""Semantic and perceptual quality gate: CLIP cosine + SSIM + PSNR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class VerificationResult:
    clip_cosine: float
    ssim: float
    psnr_db: float
    passed: bool
    reason: str  # why the gate passed or failed
    fallback_used: bool = False


class SemanticVerifier:
    """Verifies that adversarial protection preserves semantic content.

    Gate order (fail-fast):
    1. SSIM ≥ ssim_threshold (default 0.95)
    2. PSNR ≥ psnr_threshold (default 35 dB)
    3. CLIP cosine ≥ clip_threshold (default 0.92)
    """

    def __init__(
        self,
        ssim_threshold: float = 0.95,
        psnr_threshold: float = 35.0,
        clip_threshold: float = 0.92,
        hf_manager=None,
    ) -> None:
        self._ssim_t = ssim_threshold
        self._psnr_t = psnr_threshold
        self._clip_t = clip_threshold
        self._hf = hf_manager

    def verify(
        self, original: Image.Image, protected: Image.Image
    ) -> VerificationResult:
        """Run all quality gates. Returns VerificationResult with pass/fail details."""
        ssim_val, psnr_val = self._compute_structural_metrics(original, protected)

        if ssim_val < self._ssim_t:
            return VerificationResult(
                clip_cosine=0.0,
                ssim=ssim_val,
                psnr_db=psnr_val,
                passed=False,
                reason=f"SSIM {ssim_val:.4f} < threshold {self._ssim_t} — reduce epsilon",
            )

        if psnr_val < self._psnr_t:
            return VerificationResult(
                clip_cosine=0.0,
                ssim=ssim_val,
                psnr_db=psnr_val,
                passed=False,
                reason=f"PSNR {psnr_val:.2f} dB < threshold {self._psnr_t} dB — reduce epsilon",
            )

        clip_cosine, fallback = self._compute_clip_cosine(original, protected)

        if clip_cosine < self._clip_t:
            return VerificationResult(
                clip_cosine=clip_cosine,
                ssim=ssim_val,
                psnr_db=psnr_val,
                passed=False,
                reason=f"CLIP cosine {clip_cosine:.4f} < threshold {self._clip_t} — semantic drift too high",
                fallback_used=fallback,
            )

        return VerificationResult(
            clip_cosine=clip_cosine,
            ssim=ssim_val,
            psnr_db=psnr_val,
            passed=True,
            reason="All gates passed",
            fallback_used=fallback,
        )

    def verify_recovery(
        self,
        original: Image.Image,
        recovered: Image.Image,
        edited_mask: Optional[Image.Image] = None,
    ) -> VerificationResult:
        """Verify quality of recovered image (after noise reversal).

        If edited_mask is provided (white = edited region, black = unedited),
        only check unedited regions for losslessness.
        """
        if edited_mask is not None:
            import numpy as np
            mask_arr = np.array(edited_mask.convert("L"))
            unedited = mask_arr < 128
            orig_arr = np.array(original.convert("RGB")).astype(float)
            rec_arr = np.array(recovered.convert("RGB")).astype(float)
            if unedited.any():
                orig_unedited = orig_arr[unedited]
                rec_unedited = rec_arr[unedited]
                mae = float(np.abs(orig_unedited - rec_unedited).mean())
                passed = mae <= 2.0
                return VerificationResult(
                    clip_cosine=1.0,
                    ssim=1.0 - mae / 255.0,
                    psnr_db=self._psnr_float(orig_arr, rec_arr),
                    passed=passed,
                    reason=f"Unedited MAE={mae:.2f} {'≤' if passed else '>'} 2.0",
                )

        ssim_val, psnr_val = self._compute_structural_metrics(original, recovered)
        passed = ssim_val >= 0.99 and psnr_val >= 45.0
        return VerificationResult(
            clip_cosine=1.0,
            ssim=ssim_val,
            psnr_db=psnr_val,
            passed=passed,
            reason=f"Recovery SSIM={ssim_val:.4f} PSNR={psnr_val:.2f}dB",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_structural_metrics(
        original: Image.Image, protected: Image.Image
    ) -> tuple[float, float]:
        import numpy as np

        orig_arr = np.array(original.convert("RGB")).astype(float)
        prot_arr = np.array(protected.convert("RGB")).astype(float)

        psnr = SemanticVerifier._psnr_float(orig_arr, prot_arr)

        try:
            from skimage.metrics import structural_similarity as ssim
            orig_gray = orig_arr.mean(axis=2)
            prot_gray = prot_arr.mean(axis=2)
            ssim_val = float(ssim(orig_gray, prot_gray, data_range=255.0))
        except ImportError:
            mu1, mu2 = orig_arr.mean(), prot_arr.mean()
            s1, s2 = orig_arr.std(), prot_arr.std()
            s12 = float(np.mean((orig_arr - mu1) * (prot_arr - mu2)))
            C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
            ssim_val = float(
                (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
                / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 ** 2 + s2 ** 2 + C2))
            )
        return ssim_val, psnr

    @staticmethod
    def _psnr_float(original: "np.ndarray", compressed: "np.ndarray") -> float:
        import numpy as np
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return float("inf")
        return float(20 * np.log10(255.0 / np.sqrt(mse)))

    def _compute_clip_cosine(
        self, original: Image.Image, protected: Image.Image
    ) -> tuple[float, bool]:
        """Return (cosine, fallback_used). Falls back to pixel similarity if CLIP unavailable."""
        if self._hf is not None:
            try:
                cosine = self._hf.clip_similarity(original, protected)
                return cosine, False
            except Exception:
                pass

        # Heuristic fallback: pixel-level mean absolute difference normalized
        import numpy as np
        a = np.array(original.convert("RGB")).astype(float)
        b = np.array(protected.convert("RGB")).astype(float)
        diff = np.abs(a - b).mean()
        cosine = float(max(0.0, 1.0 - diff / (self._ssim_t * 10)))
        return cosine, True
