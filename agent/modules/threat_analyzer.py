"""LLM-powered privacy threat modeling for images."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

THREAT_ANALYSIS_PROMPT = """\
You are a privacy security expert specializing in AI and image data protection.
Analyze the following image metadata and characteristics:

IMAGE DESCRIPTION: {image_description}
EXIF METADATA PRESENT: {has_exif}
GPS IN EXIF: {has_gps}
FACE COUNT: {face_count}
VISIBLE TEXT DETECTED: {has_text}
IMAGE SIZE: {width}x{height}
INTENDED USE: {context}

Provide a JSON response with these EXACT fields:
{{
  "pii_types": ["list of PII types, e.g. face, name, phone, location, id_document, medical"],
  "risk_level": "low|medium|high|critical",
  "attack_scenarios": ["3 realistic attack scenarios with technical detail"],
  "recommended_epsilon": 4,
  "recommended_mode": "pixel",
  "recommendations": ["3-5 actionable privacy recommendations"],
  "confidence": 0.85
}}

risk_level rules: face=high, GPS EXIF=high, visible PII text=critical, general scene=low/medium.
recommended_epsilon: 2 for low risk, 4 for medium/high, 8 for critical.
recommended_mode: "pixel" for most cases, "combined" for critical risk.
Return only valid JSON, no markdown.
"""

PROTECTION_ADVISORY_PROMPT = """\
Given this image privacy risk assessment:
- Image type: {image_type}
- Detected PII: {pii_list}
- Risk level: {risk_level}
- Current epsilon setting: {current_epsilon}/255

Recommend the appropriate protection configuration in 2-3 sentences.
Be specific about epsilon, whether DCT mode is needed, and whether to use
Ollama (offline) instead of a cloud API. Focus on actionable advice.
"""


@dataclass
class ThreatReport:
    pii_types: list[str]
    risk_level: str  # low | medium | high | critical
    recommended_epsilon: float
    recommended_mode: str
    attack_scenarios: list[str]
    recommendations: list[str]
    confidence: float
    advisory_text: str = ""
    face_count: int = 0
    has_gps: bool = False
    has_text: bool = False


class ThreatAnalyzer:
    """Analyzes images locally and uses LLM for threat report generation.

    SAFETY GATE: The original image is NEVER sent to the LLM.
    Only metadata and a local description are sent.
    """

    def __init__(self, llm_client=None, memory=None, hf_manager=None) -> None:
        self._llm = llm_client
        self._memory = memory
        self._hf = hf_manager

    def analyze(
        self,
        image: Image.Image,
        context: str = "general image editing",
    ) -> ThreatReport:
        """Run full threat analysis pipeline. Returns ThreatReport."""
        metadata = self._extract_metadata(image)
        face_count = self._detect_faces(image)
        has_text = self._detect_text_heuristic(image)
        caption = ""
        if self._hf:
            try:
                caption = self._hf.generate_caption(image)
            except Exception:
                pass
        description = self._build_description(image, metadata, face_count, has_text, context, caption)

        if self._llm:
            report = self._llm_threat_analysis(description, metadata, face_count, has_text, context)
        else:
            report = self._heuristic_threat_analysis(metadata, face_count, has_text)

        if self._memory and report:
            try:
                self._memory.save_threat_report(
                    session_id=None,
                    pii_types=report.pii_types,
                    risk_level=report.risk_level,
                    recommended_epsilon=report.recommended_epsilon,
                    recommendations=report.recommendations,
                    attack_scenarios=report.attack_scenarios,
                    confidence=report.confidence,
                )
            except Exception:
                pass

        return report

    # ------------------------------------------------------------------
    # Local analysis helpers (never send image to LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metadata(image: Image.Image) -> dict:
        meta = {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "has_exif": False,
            "has_gps": False,
        }
        try:
            exif_data = image._getexif()  # type: ignore
            if exif_data:
                meta["has_exif"] = True
                # GPS IFD tag is 34853
                if 34853 in exif_data:
                    meta["has_gps"] = True
        except Exception:
            pass
        return meta

    @staticmethod
    def _detect_faces(image: Image.Image) -> int:
        """Detect faces locally using OpenCV haarcascade (no LLM)."""
        try:
            import cv2
            import numpy as np
            arr = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            return len(faces) if len(faces) > 0 else 0
        except Exception:
            return 0

    @staticmethod
    def _detect_text_heuristic(image: Image.Image) -> bool:
        """Heuristic: high edge density in small regions suggests text."""
        try:
            import numpy as np
            arr = np.array(image.convert("L")).astype(float)
            # Sobel approximation
            gx = np.abs(np.diff(arr, axis=1)).mean()
            gy = np.abs(np.diff(arr, axis=0)).mean()
            edge_density = (gx + gy) / 2.0
            return edge_density > 15.0
        except Exception:
            return False

    @staticmethod
    def _build_description(
        image: Image.Image,
        metadata: dict,
        face_count: int,
        has_text: bool,
        context: str,
        caption: str = "",
    ) -> str:
        parts = [
            f"{image.width}x{image.height} {image.mode} image",
            f"intended for: {context}",
        ]
        if caption:
            parts.append(f"auto-caption: {caption}")
        if face_count > 0:
            parts.append(f"{face_count} human face(s) detected")
        if metadata.get("has_gps"):
            parts.append("GPS location data in EXIF metadata")
        elif metadata.get("has_exif"):
            parts.append("EXIF metadata present (no GPS)")
        if has_text:
            parts.append("text or document content detected")
        return "; ".join(parts)

    # ------------------------------------------------------------------
    # LLM analysis
    # ------------------------------------------------------------------

    def _llm_threat_analysis(
        self,
        description: str,
        metadata: dict,
        face_count: int,
        has_text: bool,
        context: str,
    ) -> ThreatReport:
        prompt = THREAT_ANALYSIS_PROMPT.format(
            image_description=description,
            has_exif=metadata.get("has_exif", False),
            has_gps=metadata.get("has_gps", False),
            face_count=face_count,
            has_text=has_text,
            width=metadata.get("width", 0),
            height=metadata.get("height", 0),
            context=context,
        )
        try:
            raw = self._llm.complete_sync(prompt, max_tokens=1024, task="threat_analysis")
            data = json.loads(raw.strip())
            report = ThreatReport(
                pii_types=data.get("pii_types", []),
                risk_level=data.get("risk_level", "medium"),
                recommended_epsilon=float(data.get("recommended_epsilon", 4)),
                recommended_mode=data.get("recommended_mode", "pixel"),
                attack_scenarios=data.get("attack_scenarios", []),
                recommendations=data.get("recommendations", []),
                confidence=float(data.get("confidence", 0.7)),
                face_count=face_count,
                has_gps=metadata.get("has_gps", False),
                has_text=has_text,
            )
        except Exception:
            report = self._heuristic_threat_analysis(metadata, face_count, has_text)

        advisory_prompt = PROTECTION_ADVISORY_PROMPT.format(
            image_type=f"{metadata.get('width')}x{metadata.get('height')} image",
            pii_list=", ".join(report.pii_types) if report.pii_types else "none detected",
            risk_level=report.risk_level,
            current_epsilon=int(report.recommended_epsilon),
        )
        try:
            report.advisory_text = self._llm.complete_sync(
                advisory_prompt, max_tokens=256, task="advisory"
            )
        except Exception:
            report.advisory_text = self._fallback_advisory(report)

        return report

    @staticmethod
    def _heuristic_threat_analysis(
        metadata: dict, face_count: int, has_text: bool
    ) -> ThreatReport:
        pii_types = []
        risk_score = 0

        if face_count > 0:
            pii_types.append("face")
            risk_score += 2
        if metadata.get("has_gps"):
            pii_types.append("location")
            risk_score += 2
        if has_text:
            pii_types.append("visible_text")
            risk_score += 1
        if metadata.get("has_exif"):
            pii_types.append("exif_metadata")
            risk_score += 1

        if risk_score == 0:
            risk_level, epsilon = "low", 2.0
        elif risk_score <= 2:
            risk_level, epsilon = "medium", 4.0
        elif risk_score <= 4:
            risk_level, epsilon = "high", 4.0
        else:
            risk_level, epsilon = "critical", 8.0

        attack_scenarios = [
            "API provider logs raw request bytes — image stored in plaintext server-side",
            "Network-level TLS interception by corporate proxy or malicious Wi-Fi endpoint",
            "Model inversion attack: adversary reconstructs image from model training data if provider uses API logs for fine-tuning",
        ]

        recommendations = [
            f"Use epsilon={int(epsilon)}/255 noise protection for this risk level",
            "Prefer Ollama offline mode for critical privacy images (faces, documents)",
            "Strip EXIF metadata before sending even with proxy protection",
            "Use PNG encoding (lossless) to ensure exact noise reversal",
        ]
        if metadata.get("has_gps"):
            recommendations.insert(0, "CRITICAL: Remove GPS EXIF before any cloud upload")

        return ThreatReport(
            pii_types=pii_types,
            risk_level=risk_level,
            recommended_epsilon=epsilon,
            recommended_mode="combined" if risk_level == "critical" else "pixel",
            attack_scenarios=attack_scenarios,
            recommendations=recommendations,
            confidence=0.75,
            advisory_text="",
            face_count=face_count,
            has_gps=metadata.get("has_gps", False),
            has_text=has_text,
        )

    @staticmethod
    def _fallback_advisory(report: ThreatReport) -> str:
        lines = [
            f"Risk level: {report.risk_level}. ",
            f"Recommended epsilon: {int(report.recommended_epsilon)}/255. ",
        ]
        if report.has_gps:
            lines.append("Strip GPS EXIF before uploading. ")
        if report.risk_level == "critical":
            lines.append("Use offline Ollama mode to avoid all cloud data exposure. ")
        else:
            lines.append("Current protection level is sufficient for this use case.")
        return "".join(lines)
