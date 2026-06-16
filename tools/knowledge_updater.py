"""Research paper crawler: ArXiv cs.CR + cs.CV -> SECOND-KNOWLEDGE-BRAIN.md."""

from __future__ import annotations

import hashlib
import urllib.parse
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

BRAIN_PATH = Path(__file__).parent.parent / "SECOND-KNOWLEDGE-BRAIN.md"

ARXIV_CATEGORIES = ["cs.CR", "cs.CV"]
ARXIV_QUERIES = [
    "adversarial examples image privacy protection",
    "image obfuscation reversible perturbation imperceptible",
    "model inversion attack defense image",
    "face privacy protection adversarial cloaking",
    "differential privacy computer vision image noise",
]

S2_QUERIES = [
    "adversarial image privacy protection",
    "pixel space perturbation reversible privacy",
    "LLM vision API privacy defense",
    "image steganography privacy security",
    "perceptual hash image protection",
]

DOMAIN_KEYWORDS = [
    "adversarial", "privacy", "perturbation", "obfuscation",
    "imperceptible", "reversible", "image protection", "pixel noise",
    "DCT masking", "FGSM", "PGD", "cloaking", "model inversion",
    "membership inference", "face privacy",
]


@dataclass
class PaperEntry:
    title: str
    authors: str
    year: int
    venue: str
    url: str
    abstract: str
    key_finding: str = ""
    relevance: str = ""
    hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        raw = f"{self.title}{self.authors}{self.year}"
        self.hash = hashlib.sha256(raw.encode()).hexdigest()[:16]


class KnowledgeUpdater:
    def __init__(self, memory=None) -> None:
        self._memory = memory

    async def run_weekly_update(self) -> int:
        entries = await self._crawl_arxiv()
        entries += await self._crawl_semantic_scholar()
        unique = self._deduplicate(entries)
        scored = self._score_papers(unique)
        top = sorted(scored, key=lambda x: x[1], reverse=True)[:20]
        added = self._append_to_brain([p for p, _ in top])
        self._log_update(added)
        print(f"[KnowledgeUpdater] Added {added} new entries. Next run: weekly Sunday 02:00")
        return added

    async def _crawl_arxiv(self) -> list[PaperEntry]:
        entries: list[PaperEntry] = []
        cats = "|".join(ARXIV_CATEGORIES)
        headers = {"User-Agent": "ImagePrivacyAgent/1.0"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in ARXIV_QUERIES:
                try:
                    url = (
                        f"https://export.arxiv.org/api/query"
                        f"?search_query=all:{urllib.parse.quote(query)}"
                        f"&cat={cats}"
                        f"&max_results=10&sortBy=submittedDate&sortOrder=descending"
                    )
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    entries.extend(self._parse_arxiv_xml(resp.text))
                except Exception as exc:
                    print(f"[KnowledgeUpdater] ArXiv error ({query}): {exc}")
        return entries

    @staticmethod
    def _parse_arxiv_xml(xml_data: str) -> list[PaperEntry]:
        entries = []
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(xml_data)
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                published_el = entry.find("atom:published", ns)
                id_el = entry.find("atom:id", ns)
                authors = [
                    a.find("atom:name", ns).text
                    for a in entry.findall("atom:author", ns)
                    if a.find("atom:name", ns) is not None
                ]
                if not title_el or not id_el:
                    continue
                year = int(published_el.text[:4]) if published_el is not None else 2024
                abstract = (summary_el.text or "").strip().replace("\n", " ")[:300]
                entries.append(PaperEntry(
                    title=title_el.text.strip().replace("\n", " "),
                    authors=", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
                    year=year,
                    venue="ArXiv",
                    url=id_el.text.strip() if id_el.text else "",
                    abstract=abstract,
                    key_finding=abstract[:120] + "...",
                    relevance="Adversarial image privacy",
                ))
        except ET.ParseError:
            pass
        return entries

    async def _crawl_semantic_scholar(self) -> list[PaperEntry]:
        entries: list[PaperEntry] = []
        headers = {"User-Agent": "ImagePrivacyAgent/1.0"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in S2_QUERIES:
                try:
                    url = (
                        f"https://api.semanticscholar.org/graph/v1/paper/search"
                        f"?query={urllib.parse.quote(query)}"
                        f"&fields=title,authors,year,venue,externalIds,abstract"
                        f"&limit=10"
                    )
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    for paper in data.get("data", []):
                        entries.append(self._s2_to_paper_entry(paper))
                except Exception as exc:
                    print(f"[KnowledgeUpdater] S2 error ({query}): {exc}")
        return entries

    @staticmethod
    def _s2_to_paper_entry(paper: dict) -> PaperEntry:
        ext_ids = paper.get("externalIds") or {}
        doi = ext_ids.get("DOI", "")
        arxiv_id = ext_ids.get("ArXiv", "")
        url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (f"https://doi.org/{doi}" if doi else "")
        authors_raw = paper.get("authors") or []
        authors = ", ".join(a.get("name", "") for a in authors_raw[:3])
        if len(authors_raw) > 3:
            authors += " et al."
        abstract = (paper.get("abstract") or "")[:300].replace("\n", " ")
        return PaperEntry(
            title=paper.get("title", "Unknown"),
            authors=authors,
            year=paper.get("year") or 2024,
            venue=paper.get("venue", "Unknown"),
            url=url,
            abstract=abstract,
            key_finding=abstract[:120] + "...",
            relevance="Image privacy security",
        )

    def _deduplicate(self, entries: list[PaperEntry]) -> list[PaperEntry]:
        seen: set[str] = set()
        if self._memory:
            try:
                seen.update(self._memory.get_known_paper_hashes())
            except Exception:
                pass
        unique = []
        for e in entries:
            if e.hash not in seen:
                seen.add(e.hash)
                unique.append(e)
        return unique

    @staticmethod
    def _score_papers(entries: list[PaperEntry]) -> list[tuple[PaperEntry, float]]:
        now_year = datetime.now(timezone.utc).year
        scored = []
        for e in entries:
            recency = max(0.0, 1.0 - (now_year - e.year) / 5.0)
            combined = (e.title + " " + e.abstract).lower()
            hits = sum(1 for kw in DOMAIN_KEYWORDS if kw.lower() in combined)
            relevance = min(1.0, hits / 5.0)
            score = 0.6 * recency + 0.4 * relevance
            scored.append((e, score))
        return scored

    def _append_to_brain(self, entries: list[PaperEntry]) -> int:
        added = 0
        brain = BRAIN_PATH.read_text(encoding="utf-8") if BRAIN_PATH.exists() else ""

        for entry in entries:
            if entry.url in brain or entry.title[:40] in brain:
                continue
            count = len(re.findall(r'^\\| \\d+', brain, re.MULTILINE)) + 1
            row = (
                f"| {count} "
                f"| {entry.title[:60]} | {entry.authors} | {entry.year} | {entry.venue} "
                f"| {entry.url} | {entry.key_finding[:80]} | {entry.relevance} |\n"
            )
            with BRAIN_PATH.open("a", encoding="utf-8") as f:
                f.write(row)
            if self._memory:
                try:
                    self._memory.mark_paper_known(entry.hash)
                except Exception:
                    pass
            added += 1
        return added

    def _log_update(self, added: int) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_line = f"| {now} | ArXiv+S2 | {added} | — | Weekly auto-update |\n"
        if BRAIN_PATH.exists():
            content = BRAIN_PATH.read_text(encoding="utf-8")
            if "## Knowledge Update Log" in content:
                updated = content.replace(
                    "| Date | Source |",
                    f"{log_line}| Date | Source |",
                )
                BRAIN_PATH.write_text(updated, encoding="utf-8")

    def start_scheduled(self) -> None:
        """Start APScheduler weekly cron. Call from orchestrator startup."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger
            scheduler = AsyncIOScheduler()
            scheduler.add_job(
                self.run_weekly_update,
                CronTrigger(day_of_week="sun", hour=2, minute=0),
                id="knowledge_weekly",
                replace_existing=True,
            )
            scheduler.start()
        except ImportError:
            print("[KnowledgeUpdater] APScheduler not installed — scheduled updates disabled.")


if __name__ == "__main__":
    import asyncio
    updater = KnowledgeUpdater()
    added = asyncio.run(updater.run_weekly_update())
    print(f"Done. {added} new papers added.")
