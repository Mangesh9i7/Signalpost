from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_env(env_path: Path | str | None = None) -> dict[str, str]:
    """
    Zero-dependency .env loader that populates os.environ without overwriting existing vars.
    Checks repo root and current working directory.
    """
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    else:
        candidates.extend([
            ROOT_DIR / ".env",
            Path.cwd() / ".env",
        ])

    loaded: dict[str, str] = {}
    for candidate in candidates:
        if candidate.is_file():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
                            loaded[key] = val
            except Exception:
                pass
            break
    return loaded


# Automatically load .env on import if available
load_env()


def get_llm_config() -> dict[str, Any] | None:
    """
    Detects if an LLM provider key is available in the environment.
    Supported: OpenAI / compatible, Gemini / Google, Anthropic, Groq.
    """
    # 1. OpenAI / OpenAI-compatible endpoint
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    custom_model = os.environ.get("LLM_MODEL")

    if openai_key or base_url:
        endpoint = f"{base_url}/chat/completions" if base_url else "https://api.openai.com/v1/chat/completions"
        model = custom_model or ("gpt-4o-mini" if "openai.com" in endpoint or not base_url else "default")
        return {
            "provider": "openai_compatible",
            "api_key": openai_key or "no-key",
            "endpoint": endpoint,
            "model": model,
        }

    # 2. Google Gemini
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        model = custom_model or "gemini-1.5-flash"
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        return {
            "provider": "gemini",
            "api_key": gemini_key,
            "endpoint": endpoint,
            "model": model,
        }

    # 3. Groq
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        return {
            "provider": "openai_compatible",
            "api_key": groq_key,
            "endpoint": "https://api.groq.com/openai/v1/chat/completions",
            "model": custom_model or "llama-3.3-70b-versatile",
        }

    # 4. Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        return {
            "provider": "anthropic",
            "api_key": anthropic_key,
            "endpoint": "https://api.anthropic.com/v1/messages",
            "model": custom_model or "claude-3-5-haiku-latest",
        }

    return None


def is_llm_available() -> bool:
    """Returns True if an LLM is configured and ready to use, False otherwise."""
    return get_llm_config() is not None


def call_llm(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 500,
    temperature: float = 0.1,
    timeout: float = 12.0,
) -> str | None:
    """
    Executes a chat completion call against the configured LLM provider.
    Fails gracefully to None on any network or auth error so the pipeline never breaks.
    """
    config = get_llm_config()
    if not config:
        return None

    provider = config["provider"]
    endpoint = config["endpoint"]
    model = config["model"]
    api_key = config["api_key"]

    try:
        if provider == "openai_compatible":
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "SignalpostResearchAgent/1.0",
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()

        elif provider == "gemini":
            # Gemini contents format
            contents = []
            for msg in messages:
                role = "user" if msg["role"] in {"user", "system"} else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            payload = {
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            }
            headers = {"Content-Type": "application/json"}
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
                return None

        elif provider == "anthropic":
            system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_msgs = [m for m in messages if m["role"] != "system"]
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": user_msgs,
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "User-Agent": "SignalpostResearchAgent/1.0",
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                for block in result.get("content", []):
                    if block.get("type") == "text":
                        return block.get("text", "").strip()
                return None

    except Exception:
        # Intentionally swallow errors so the batch orchestrator never fails
        return None

    return None


def generate_company_summary_with_llm(profile: dict[str, Any]) -> str | None:
    """
    Uses LLM to synthesize a concise, evidence-grounded company summary.
    If LLM is unavailable, returns a deterministic summary template.
    """
    name = profile.get("name", "Unknown Company")
    org = profile.get("organisation_number", "")
    form = profile.get("legal_form", "")
    municipality = profile.get("municipality", "")
    employees = profile.get("employees")

    evidence_dict = profile.get("evidence", {})
    fin_val = (evidence_dict.get("financials", {}).get("value") or {}).get("records", [])
    latest_fin = fin_val[0] if fin_val else {}
    revenue = latest_fin.get("revenue")
    operating_result = latest_fin.get("operating_result")

    roles_list = (evidence_dict.get("roles", {}).get("value") or {}).get("roles", [])
    active_roles = [r for r in roles_list if not r.get("inactive")]
    ceo = next((r.get("name") for r in active_roles if r.get("role_code") == "DAGL"), None)
    chair = next((r.get("name") for r in active_roles if r.get("role_code") == "LEDE"), None)

    web_val = evidence_dict.get("website", {}).get("value") or {}
    web_desc = web_val.get("description") or ""

    # Check if LLM is configured
    if is_llm_available():
        context = {
            "name": name,
            "organisation_number": org,
            "legal_form": form,
            "municipality": municipality,
            "employees": employees,
            "annual_revenue_nok": revenue,
            "operating_result_nok": operating_result,
            "chief_executive": ceo,
            "board_chair": chair,
            "website_description": web_desc,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Norwegian commercial intelligence analyst. "
                    "Write a concise, professional 2-sentence executive summary of the Norwegian company "
                    "grounded strictly in the provided verified registry and financial facts. "
                    "Never invent or speculate. If a metric is unknown, do not mention it."
                ),
            },
            {
                "role": "user",
                "content": f"Company profile data:\n{json.dumps(context, ensure_ascii=False, indent=2)}",
            },
        ]
        summary = call_llm(messages, max_tokens=150, temperature=0.1)
        if summary:
            return summary

    # Deterministic fallback when LLM is not configured or fails
    parts = [f"{name} ({org}) is an active {form or 'company'} based in {municipality or 'Norway'}."]
    if employees is not None:
        parts.append(f"It registers {employees} employees.")
    if revenue is not None:
        parts.append(f"Latest reported annual revenue is {revenue:,.0f} NOK.")
    if ceo:
        parts.append(f"Led by {ceo}.")
    return " ".join(parts)


def answer_question_with_llm(facts: list[dict[str, Any]], question: str, company_name: str) -> str | None:
    """
    Synthesizes a natural language answer to a user question using verified facts as context.
    """
    if not is_llm_available() or not facts:
        return None

    facts_summary = "\n".join(f"- {f.get('claim')}: {f.get('value')} (Source: {f.get('source_url')})" for f in facts)
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a factual business intelligence assistant researching {company_name}. "
                "Answer the user's question directly using ONLY the provided verified facts. "
                "Cite relevant figures and names. If the facts do not contain the answer, say what is known and what is missing."
            ),
        },
        {
            "role": "user",
            "content": f"Verified facts:\n{facts_summary}\n\nQuestion: {question}",
        },
    ]
    return call_llm(messages, max_tokens=250, temperature=0.1)
