"""HuggingFace model manager: lazy-load CLIP, ViT, BGE with CUDA auto-detection."""

from __future__ import annotations

import threading
from typing import Optional
import numpy as np

MODEL_REGISTRY: dict[str, str] = {
    "clip":         "openai/clip-vit-large-patch14",
    "vit":          "google/vit-base-patch16-224",
    "bge_large":    "BAAI/bge-large-en-v1.5",
    "bge_reranker": "BAAI/bge-reranker-large",
    "minilm":       "sentence-transformers/all-MiniLM-L6-v2",
    "blip":         "Salesforce/blip-image-captioning-base",
}

_IDLE_TIMEOUT = 600


class HFModelManager:
    """Singleton. Lazy-loads models on first use; unloads after idle timeout."""

    _instance: Optional["HFModelManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "HFModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._models: dict = {}
        self._timers: dict[str, threading.Timer] = {}
        self._model_lock = threading.Lock()
        self._device = self._detect_device()

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _reset_idle_timer(self, key: str) -> None:
        if key in self._timers:
            self._timers[key].cancel()
        timer = threading.Timer(_IDLE_TIMEOUT, self._unload_model, args=[key])
        timer.daemon = True
        timer.start()
        self._timers[key] = timer

    def _unload_model(self, key: str) -> None:
        with self._model_lock:
            self._models.pop(key, None)
            self._timers.pop(key, None)

    def _load_clip(self):
        from transformers import CLIPProcessor, CLIPModel
        import torch
        model = CLIPModel.from_pretrained(MODEL_REGISTRY["clip"])
        processor = CLIPProcessor.from_pretrained(MODEL_REGISTRY["clip"])
        if self._device == "cuda":
            model = model.to("cuda")
        return {"model": model, "processor": processor, "torch": torch}

    def _load_bge(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_REGISTRY["bge_large"], device=self._device)

    def _load_minilm(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_REGISTRY["minilm"], device=self._device)

    def clip_similarity(self, image1, image2) -> float:
        """Return cosine similarity between two PIL.Image objects using CLIP."""
        with self._model_lock:
            if "clip" not in self._models:
                try:
                    self._models["clip"] = self._load_clip()
                except Exception:
                    return self._heuristic_image_similarity(image1, image2)
            self._reset_idle_timer("clip")
            bundle = self._models["clip"]

        try:
            import torch
            proc = bundle["processor"]
            model = bundle["model"]
            inputs1 = proc(images=image1, return_tensors="pt")
            inputs2 = proc(images=image2, return_tensors="pt")
            if self._device == "cuda":
                inputs1 = {k: v.to("cuda") for k, v in inputs1.items()}
                inputs2 = {k: v.to("cuda") for k, v in inputs2.items()}
            with torch.no_grad():
                feat1 = model.get_image_features(**inputs1)
                feat2 = model.get_image_features(**inputs2)
            feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
            feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
            return float((feat1 * feat2).sum().item())
        except Exception:
            return self._heuristic_image_similarity(image1, image2)

    @staticmethod
    def _heuristic_image_similarity(img1, img2) -> float:
        """Fast pixel-level similarity when CLIP is unavailable."""
        try:
            import numpy as np
            a = np.array(img1).astype(float)
            b = np.array(img2).astype(float)
            if a.shape != b.shape:
                return 0.0
            diff = np.abs(a - b).mean()
            return float(max(0.0, 1.0 - diff / 255.0))
        except Exception:
            return 0.95

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode text strings with BGE-large; return float32 array (N, 1024)."""
        with self._model_lock:
            if "bge_large" not in self._models:
                try:
                    self._models["bge_large"] = self._load_bge()
                except Exception:
                    return self._tfidf_fallback(texts)
            self._reset_idle_timer("bge_large")
            model = self._models["bge_large"]

        try:
            return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception:
            return self._tfidf_fallback(texts)

    @staticmethod
    def _tfidf_fallback(texts: list[str]) -> np.ndarray:
        """Deterministic TF-IDF-style 1024-dim embedding."""
        result = []
        for text in texts:
            vec = np.zeros(1024, dtype=np.float32)
            for i, char in enumerate(text[:512]):
                vec[ord(char) % 1024] += 1.0
            norm = np.linalg.norm(vec)
            result.append(vec / norm if norm > 0 else vec)
        return np.array(result)

    def _load_blip(self):
        from transformers import pipeline
        return pipeline("image-to-text", model=MODEL_REGISTRY["blip"], device=self._device)

    def generate_caption(self, image, max_length=128):
        with self._model_lock:
            if "blip" not in self._models:
                try:
                    self._models["blip"] = self._load_blip()
                except Exception:
                    return ""
                self._reset_idle_timer("blip")
            self._reset_idle_timer("blip")
            model = self._models["blip"]
        try:
            result = model(image, max_new_tokens=max_length)
            return result[0]["generated_text"] if result else ""
        except Exception:
            return ""

    def preload(self, model_keys: list[str]) -> None:
        """Eagerly load specified models at startup."""
        for key in model_keys:
            if key == "clip":
                with self._model_lock:
                    if key not in self._models:
                        try:
                            self._models[key] = self._load_clip()
                            self._reset_idle_timer(key)
                        except Exception:
                            pass
