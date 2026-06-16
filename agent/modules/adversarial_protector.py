"""Core privacy transform engine: keyed pixel-space noise + DCT block masking."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class ProtectionStats:
    l_inf_norm: float
    l2_norm: float
    psnr_db: float
    ssim: float
    max_pixel_diff: int
    protection_mode: str  # "pixel", "dct", "combined"


_MASTER_SECRET = os.getenv(
    "PRIVACY_MASTER_SECRET",
    "image-privacy-agent-default-secret-change-in-production",
)


def _derive_session_key(session_id: str) -> bytes:
    """PBKDF2-derive a 32-byte key from session_id + master secret."""
    import hashlib
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        session_id.encode(),
        _MASTER_SECRET.encode(),
        iterations=100_000,
        dklen=32,
    )
    return dk


class AdversarialProtector:
    """Applies and reverses session-keyed imperceptible noise on images.

    Two protection modes:
    - "pixel": Uniform L∞ noise derived from session key via PCG64 PRNG.
               ε ≤ epsilon_int gray levels (default 4 → 4/255 = 0.016 L∞).
    - "combined": Pixel noise + 8×8 block DCT coefficient zeroing for
                  defense-in-depth against frequency-domain attacks.
    """

    def __init__(self, epsilon_int: int = 4, mode: str = "pixel") -> None:
        if epsilon_int < 1 or epsilon_int > 32:
            raise ValueError("epsilon_int must be in [1, 32]")
        self._epsilon_int = epsilon_int
        self._mode = mode
        self._masks: dict[str, np.ndarray] = {}
        self._dct_masks: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def protect(self, image: Image.Image, session_id: str) -> Image.Image:
        """Add keyed noise to image. Store mask for later reversal."""
        arr = np.array(image.convert("RGB")).astype(np.int32)
        key = _derive_session_key(session_id)
        rng = np.random.Generator(np.random.PCG64(seed=int.from_bytes(key[:8], "big")))

        noise = rng.integers(
            -self._epsilon_int,
            self._epsilon_int + 1,
            size=arr.shape,
            dtype=np.int16,
        )
        self._masks[session_id] = noise

        protected = np.clip(arr + noise.astype(np.int32), 0, 255).astype(np.uint8)

        if self._mode == "combined":
            protected, dct_mask = self._apply_dct_mask(protected, key)
            self._dct_masks[session_id] = dct_mask

        result = Image.fromarray(protected, mode="RGB")
        if image.mode == "RGBA":
            r, g, b = result.split()
            a = image.split()[3]
            result = Image.merge("RGBA", (r, g, b, a))
        return result

    def reverse(self, image: Image.Image, session_id: str) -> Image.Image:
        """Remove noise using stored session mask. Returns image unchanged if no mask."""
        if session_id not in self._masks:
            return image

        mode = image.mode
        arr = np.array(image.convert("RGB")).astype(np.int32)

        if self._mode == "combined" and session_id in self._dct_masks:
            arr = self._reverse_dct_mask(arr.astype(np.uint8), self._dct_masks.pop(session_id))
            arr = arr.astype(np.int32)

        noise = self._masks.pop(session_id).astype(np.int32)
        recovered = np.clip(arr - noise, 0, 255).astype(np.uint8)

        result = Image.fromarray(recovered, mode="RGB")
        if mode == "RGBA":
            r, g, b = result.split()
            a = image.split()[3]
            result = Image.merge("RGBA", (r, g, b, a))
        return result

    def compute_stats(
        self, original: Image.Image, protected: Image.Image
    ) -> ProtectionStats:
        """Compute imperceptibility metrics for the protect transform."""
        o = np.array(original.convert("RGB")).astype(float)
        p = np.array(protected.convert("RGB")).astype(float)
        noise = p - o
        ssim_val = self._compute_ssim(o, p)
        return ProtectionStats(
            l_inf_norm=float(np.abs(noise).max()),
            l2_norm=float(np.linalg.norm(noise)),
            psnr_db=self._psnr(o, p),
            ssim=ssim_val,
            max_pixel_diff=int(np.abs(noise).max()),
            protection_mode=self._mode,
        )

    # ------------------------------------------------------------------
    # DCT helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_dct_mask(
        arr: np.ndarray, key: bytes, block_size: int = 8, zeroed_coeffs: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        """Zero top-k highest-frequency DCT coefficients in each 8×8 block.

        Returns modified image array and a binary mask recording which coefficients
        were zeroed (for exact reversal).
        """
        try:
            from scipy.fft import dctn, idctn
        except ImportError:
            return arr, np.zeros_like(arr, dtype=np.float32)

        result = arr.copy().astype(np.float32)
        h, w, c = arr.shape
        mask = np.zeros_like(result)

        for ch in range(c):
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = result[y:y + block_size, x:x + block_size, ch].copy()
                    dct_block = dctn(block, norm="ortho")
                    # Zero the bottom-right corner (highest frequencies)
                    for i in range(block_size - 1, block_size - 1 - zeroed_coeffs, -1):
                        mask[y + i, x + i, ch] = dct_block[i, i]
                        dct_block[i, i] = 0.0
                    result[y:y + block_size, x:x + block_size, ch] = idctn(dct_block, norm="ortho")

        return np.clip(result, 0, 255).astype(np.uint8), mask

    @staticmethod
    def _reverse_dct_mask(
        arr: np.ndarray, mask: np.ndarray, block_size: int = 8
    ) -> np.ndarray:
        """Restore zeroed DCT coefficients from stored mask."""
        try:
            from scipy.fft import dctn, idctn
        except ImportError:
            return arr

        result = arr.copy().astype(np.float32)
        h, w, c = arr.shape

        for ch in range(c):
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = result[y:y + block_size, x:x + block_size, ch].copy()
                    dct_block = dctn(block, norm="ortho")
                    for i in range(block_size):
                        if abs(mask[y + i, x + i, ch]) > 0:
                            dct_block[i, i] = mask[y + i, x + i, ch]
                    result[y:y + block_size, x:x + block_size, ch] = idctn(dct_block, norm="ortho")

        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _psnr(original: np.ndarray, compressed: np.ndarray) -> float:
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return float("inf")
        return float(20 * np.log10(255.0 / np.sqrt(mse)))

    @staticmethod
    def _compute_ssim(
        original: np.ndarray, protected: np.ndarray
    ) -> float:
        try:
            from skimage.metrics import structural_similarity as ssim
            orig_gray = original.mean(axis=2) if original.ndim == 3 else original
            prot_gray = protected.mean(axis=2) if protected.ndim == 3 else protected
            return float(ssim(orig_gray, prot_gray, data_range=255.0))
        except ImportError:
            pass
        # Fallback: simplified SSIM approximation
        mu1, mu2 = original.mean(), protected.mean()
        s1, s2 = original.std(), protected.std()
        s12 = float(np.mean((original - mu1) * (protected - mu2)))
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        return float(
            (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
            / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 ** 2 + s2 ** 2 + C2))
        )
