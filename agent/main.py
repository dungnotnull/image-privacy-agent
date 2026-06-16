"""Image Privacy Agent — CLI entry point and FastAPI proxy server."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import click


@click.group()
def cli():
    """Image Privacy Agent: protect personal images during LLM API calls."""


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host")
@click.option("--port", default=8003, show_default=True, help="Proxy port")
@click.option("--epsilon", default=4, show_default=True, help="Noise strength (1-32 gray levels)")
@click.option("--mode", default="pixel", type=click.Choice(["pixel", "combined"]), show_default=True)
@click.option("--start-scheduler/--no-scheduler", default=True)
def serve(host: str, port: int, epsilon: int, mode: str, start_scheduler: bool):
    """Start the privacy proxy server.

    Configure your LLM client:
      OPENAI_BASE_URL=http://localhost:8003/openai
      ANTHROPIC_BASE_URL=http://localhost:8003/anthropic
    """
    import uvicorn
    from agent.orchestrator import PrivacyProxyOrchestrator
    from agent.modules.proxy_interceptor import create_proxy_app

    click.echo("=" * 60)
    click.echo("  Image Privacy Agent — Local Proxy")
    click.echo("=" * 60)
    click.echo(f"  Listening on:  http://{host}:{port}")
    click.echo(f"  Epsilon:       {epsilon}/255 ({epsilon/255*100:.1f}% L∞ noise)")
    click.echo(f"  Mode:          {mode}")
    click.echo()
    click.echo("  Configure your LLM client:")
    click.echo(f"    OPENAI_BASE_URL=http://{host}:{port}/openai")
    click.echo(f"    ANTHROPIC_BASE_URL=http://{host}:{port}/anthropic")
    click.echo("=" * 60)

    orchestrator = PrivacyProxyOrchestrator(
        config={"epsilon_int": epsilon, "protection_mode": mode}
    )
    if start_scheduler:
        orchestrator.start_scheduler()

    app = create_proxy_app(orchestrator)
    uvicorn.run(app, host=host, port=port, log_level="info")


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--context", default="general image editing", help="Intended use of image")
@click.option("--output", "-o", default=None, help="Save JSON report to file")
def analyze(image_path: str, context: str, output: str):
    """Run privacy threat analysis on a local image file.

    The image is never sent to any external service — analysis is local only.
    """
    from PIL import Image
    from agent.orchestrator import PrivacyProxyOrchestrator

    click.echo(f"Analyzing: {image_path}")
    click.echo("(No external API calls — fully local analysis)")
    click.echo()

    orchestrator = PrivacyProxyOrchestrator()
    image = Image.open(image_path)
    report = orchestrator.analyze_threat(image, context)

    click.echo(f"Risk Level:        {report['risk_level'].upper()}")
    click.echo(f"PII Detected:      {', '.join(report['pii_types']) or 'none'}")
    click.echo(f"Recommended ε:     {int(report['recommended_epsilon'])}/255")
    click.echo(f"Recommended Mode:  {report['recommended_mode']}")
    click.echo(f"Confidence:        {report['confidence']:.0%}")
    click.echo()
    click.echo("Attack Scenarios:")
    for i, scenario in enumerate(report["attack_scenarios"], 1):
        click.echo(f"  {i}. {scenario}")
    click.echo()
    click.echo("Recommendations:")
    for rec in report["recommendations"]:
        click.echo(f"  • {rec}")
    if report.get("advisory"):
        click.echo()
        click.echo(f"Advisory: {report['advisory']}")

    if output:
        Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        click.echo(f"\nReport saved to: {output}")


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=".", help="Directory to save protected/recovered images")
def benchmark(image_path: str, output_dir: str):
    """Benchmark protection quality (SSIM, PSNR, CLIP) across epsilon values."""
    from PIL import Image
    from agent.orchestrator import PrivacyProxyOrchestrator

    orchestrator = PrivacyProxyOrchestrator()
    image = Image.open(image_path).convert("RGB")
    results = orchestrator.benchmark_image(image)

    click.echo(f"\nBenchmark results for: {image_path}")
    click.echo(f"{'Epsilon':>8} {'SSIM':>8} {'PSNR(dB)':>10} {'CLIP':>8} {'RecovSSIM':>10} {'Gate':>6}")
    click.echo("-" * 60)
    for label, metrics in results.items():
        eps = label.split("_")[1]
        gate = "✓" if metrics["passed_gate"] else "✗"
        click.echo(
            f"{eps:>8} {metrics['ssim']:>8.4f} {metrics['psnr_db']:>10.2f} "
            f"{metrics['clip_cosine']:>8.4f} {metrics['recovery_ssim']:>10.4f} {gate:>6}"
        )


@cli.command("update-knowledge")
def update_knowledge():
    """Manually trigger research paper crawl → SECOND-KNOWLEDGE-BRAIN.md."""
    from agent.orchestrator import PrivacyProxyOrchestrator

    orchestrator = PrivacyProxyOrchestrator()
    click.echo("Crawling ArXiv cs.CR + cs.CV and Semantic Scholar...")
    result = asyncio.run(orchestrator.update_knowledge())
    click.echo(f"Done. Papers added: {result['papers_added']}")


@cli.command("cost-report")
def cost_report():
    """Show LLM API cost breakdown for the last 30 days."""
    from agent.orchestrator import PrivacyProxyOrchestrator

    orchestrator = PrivacyProxyOrchestrator()
    report = orchestrator.get_cost_report()
    click.echo("\nLLM API Costs (last 30 days):")
    for provider, data in report["costs_30d"].items():
        click.echo(f"  {provider:<12} ${data['total_cost']:.6f}  ({data['calls']} calls)")
    click.echo()
    click.echo("Session Stats:")
    stats = report["session_stats"]
    click.echo(f"  Total sessions:  {stats.get('total_sessions', 0)}")
    click.echo(f"  Avg SSIM:        {stats.get('avg_ssim', 0):.4f}")
    click.echo(f"  Avg CLIP cosine: {stats.get('avg_clip_cosine', 0):.4f}")
    click.echo(f"  Avg PSNR:        {stats.get('avg_psnr_db', 0):.2f} dB")
    click.echo(f"  Reversals done:  {stats.get('total_reversals', 0)}")


@cli.command()
def status():
    """Show proxy runtime status and database statistics."""
    from agent.orchestrator import PrivacyProxyOrchestrator

    orchestrator = PrivacyProxyOrchestrator()
    stats = orchestrator.memory.get_stats()
    click.echo(json.dumps(stats, indent=2))


if __name__ == "__main__":
    cli()
