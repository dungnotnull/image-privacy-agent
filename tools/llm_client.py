"""Unified LLM client: Claude (primary) -> OpenAI (fallback) -> Ollama (offline)."""

from __future__ import annotations

import json
import os
from typing import AsyncGenerator, Optional, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from agent.memory.memory_manager import PrivacyMemoryManager

COST_PER_1K: dict[str, float] = {
    "claude-opus-4-8":          0.015,
    "claude-sonnet-4-6":        0.003,
    "claude-haiku-4-5-20251001": 0.00025,
    "gpt-4o":                   0.005,
    "gpt-4o-mini":              0.00015,
    "ollama/llama3":            0.0,
    "ollama/mistral":           0.0,
}

DEFAULT_MODELS = {
    "claude":  os.getenv("CLAUDE_MODEL", "claude-opus-4-8"),
    "openai":  os.getenv("OPENAI_MODEL", "gpt-4o"),
    "ollama":  os.getenv("OLLAMA_MODEL", "llama3"),
}

PROVIDER_GUIDANCE = {
    "claude":  "Long-context reasoning, threat analysis, research synthesis",
    "openai":  "Multimodal analysis, structured JSON output",
    "ollama":  "Privacy-sensitive analysis, offline/air-gapped mode",
}


class LLMClient:
    """Thread-safe unified LLM client with automatic provider fallback."""

    def __init__(self, memory: Optional["PrivacyMemoryManager"] = None):
        self._memory = memory
        self._privacy_mode = os.getenv("PRIVACY_MODE", "false").lower() == "true"

    def _build_provider_chain(self) -> list[str]:
        if self._privacy_mode:
            return ["ollama"]
        chain = []
        if os.getenv("ANTHROPIC_API_KEY"):
            chain.append("claude")
        if os.getenv("OPENAI_API_KEY"):
            chain.append("openai")
        chain.append("ollama")
        return chain or ["ollama"]

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
        task: str = "general",
    ) -> str:
        chain = self._build_provider_chain()
        last_error: Exception = RuntimeError("No providers configured")
        for provider in chain:
            try:
                result = await self._call_with_retry(provider, prompt, system, max_tokens)
                self._log_cost(provider, prompt, result, task)
                return result
            except Exception as exc:
                last_error = exc
        return f"[LLM unavailable — all providers failed. Last error: {last_error}]"

    async def stream(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        chain = self._build_provider_chain()
        for provider in chain:
            try:
                async for chunk in self._stream_provider(provider, prompt, system, max_tokens):
                    yield chunk
                return
            except Exception:
                continue
        yield "[stream unavailable]"

    async def _call_with_retry(
        self,
        provider: str,
        prompt: str,
        system: str,
        max_tokens: int,
        retries: int = 3,
    ) -> str:
        import asyncio
        for attempt in range(retries):
            try:
                return await self._call_provider(provider, prompt, system, max_tokens)
            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    async def _call_provider(
        self, provider: str, prompt: str, system: str, max_tokens: int
    ) -> str:
        if provider == "claude":
            return await self._call_claude(prompt, system, max_tokens)
        elif provider == "openai":
            return await self._call_openai(prompt, system, max_tokens)
        else:
            return await self._call_ollama(prompt, system, max_tokens)

    async def _call_claude(self, prompt: str, system: str, max_tokens: int) -> str:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        kwargs: dict = {"model": DEFAULT_MODELS["claude"], "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        msg = await client.messages.create(**kwargs)
        return msg.content[0].text

    async def _call_openai(self, prompt: str, system: str, max_tokens: int) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await client.chat.completions.create(
            model=DEFAULT_MODELS["openai"],
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    async def _call_ollama(self, prompt: str, system: str, max_tokens: int) -> str:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        payload = {
            "model": DEFAULT_MODELS["ollama"],
            "prompt": f"{system}\n\n{prompt}" if system else prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{base}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data.get("response", "")

    async def _stream_provider(
        self, provider: str, prompt: str, system: str, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        if provider == "claude":
            async for chunk in self._stream_claude(prompt, system, max_tokens):
                yield chunk
        elif provider == "openai":
            async for chunk in self._stream_openai(prompt, system, max_tokens):
                yield chunk
        else:
            result = await self._call_ollama(prompt, system, max_tokens)
            yield result

    async def _stream_claude(
        self, prompt: str, system: str, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        kwargs: dict = {"model": DEFAULT_MODELS["claude"], "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def _stream_openai(
        self, prompt: str, system: str, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        async for chunk in await client.chat.completions.create(
            model=DEFAULT_MODELS["openai"],
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        ):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def complete_sync(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        task: str = "general",
    ) -> str:
        """Synchronous wrapper around async complete()."""
        import asyncio
        coro = self.complete(prompt, system, max_tokens, task)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # Inside a running event loop: run in a fresh thread with its own loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result(timeout=120)

    def _log_cost(self, provider: str, prompt: str, result: str, task: str) -> None:
        if not self._memory:
            return
        model = DEFAULT_MODELS.get(provider, provider)
        rate = COST_PER_1K.get(model, 0.005)
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(result) // 4
        cost = (prompt_tokens + completion_tokens) / 1000 * rate
        try:
            self._memory.log_llm_cost(provider, model, prompt_tokens, completion_tokens, cost, task)
        except Exception:
            pass
