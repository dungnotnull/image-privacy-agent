"""YAML / env config loader for the image privacy agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_DEFAULT_CONFIG: dict[str, Any] = {
    "agent": {"name": "image-privacy-agent", "version": "1.0.0"},
    "server": {"host": "127.0.0.1", "port": 8003, "log_level": "info", "max_request_size_mb": 50},
    "protection": {"epsilon_int": 4, "mode": "pixel", "dct_block_size": 8, "dct_zeroed_coeffs": 3, "max_image_size_mb": 20},
    "quality_gates": {"ssim_threshold": 0.95, "psnr_threshold": 35.0, "clip_threshold": 0.92, "recovery_ssim": 0.99, "recovery_mae": 2.0},
    "llm": {"provider_priority": ["claude", "openai", "ollama"], "retry_attempts": 3, "retry_backoff_base": 1.0},
    "upstream_apis": {"request_timeout_seconds": 120},
    "memory": {"db_path": "./data/privacy_agent.db", "session_retention_days": 90, "cost_summary_days": 30},
    "threat_analyzer": {"face_detection": True, "text_detection": True, "exif_analysis": True},
}


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config from YAML then overlay env vars."""
    config = _deep_copy(_DEFAULT_CONFIG)
    cfg_file = Path(path) if path else Path(__file__).parent.parent / "config" / "agent_config.yaml"
    if cfg_file.exists():
        try:
            import yaml
            with cfg_file.open("r", encoding="utf-8") as fh:
                file_cfg = yaml.safe_load(fh) or {}
            _merge_dict(config, file_cfg)
        except ImportError:
            pass
    _overlay_env(config)
    return config


def _deep_copy(d: dict) -> dict:
    import copy
    return copy.deepcopy(d)


def _merge_dict(base: dict, overlay: dict) -> None:
    for key, val in overlay.items():
        if isinstance(val, dict) and key in base and isinstance(base[key], dict):
            _merge_dict(base[key], val)
        else:
            base[key] = val


def _overlay_env(config: dict) -> None:
    """Overlay specific env vars onto config."""
    if os.getenv("EPSILON_INT"):
        try:
            config["protection"]["epsilon_int"] = int(os.getenv("EPSILON_INT"))
        except ValueError:
            pass
    if os.getenv("PROTECTION_MODE"):
        config["protection"]["mode"] = os.getenv("PROTECTION_MODE")
    if os.getenv("PROXY_HOST"):
        config["server"]["host"] = os.getenv("PROXY_HOST")
    if os.getenv("PROXY_PORT"):
        try:
            config["server"]["port"] = int(os.getenv("PROXY_PORT"))
        except ValueError:
            pass
    if os.getenv("SSIM_THRESHOLD"):
        try:
            config["quality_gates"]["ssim_threshold"] = float(os.getenv("SSIM_THRESHOLD"))
        except ValueError:
            pass
    if os.getenv("PRIVACY_MODE", "").lower() == "true":
        config["llm"]["provider_priority"] = ["ollama"]


if __name__ == "__main__":
    import json
    print(json.dumps(load_config(), indent=2))
