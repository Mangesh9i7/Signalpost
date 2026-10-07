from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable
from urllib.parse import urlparse

from .evidence import evidence, utc_now
from .http import FetchResult, fetch_json
from .identity import apply_website_identity_gate
from .official import fetch_official_modules
from .website import fetch_website, normalize_homepage


NEWS_PATH = re.compile(r"/(?:news|press|aktuelt|nyheter|artikler|blog)(?:/|$)", re.I)


def _digest(data: str | bytes) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def extract_site_observations(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract approved external-footprint observations from an exact verified company website."""
    org = str(profile.get("organisation_number") or "")
    web_evidence = profile.get("evidence", {}).get("website", {})
    if web_evidence.get("status") != "available":
        return []

    val = web_evidence.get("value") or {}
    identity = val.get("identity_assessment") or {}
    if not identity.get("publishable"):
        return []

    observations: list[dict[str, Any]] = []
    source_url = val.get("final_url") or web_evidence.get("source_url")
    retrieved_at = web_evidence.get("retrieved_at") or utc_now()
    content_hash = str(val.get("content_sha256") or web_evidence.get("content_sha256") or _digest(source_url))
    pages = val.get("pages") or []
    social_links = val.get("social_links") or []

    # 1. Site surface completeness & metrics (permitted_public_page / approved)
    activity_proof = list(identity.get("promotion_proof") or []) + [
        {"type": "website_identity_gate", "status": identity.get("status"), "score": identity.get("score")}
    ]
    observations.append({
        "id": f"company-site-activity-{org}-{content_hash[:16]}",
        "organisation_number": org,
        "platform": "company_site",
        "signal_type": "profile_metrics",
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "content_sha256": content_hash,
        "exact_entity": True,
        "identity_proof": activity_proof,
        "acquisition_mode": "permitted_public_page",
        "rights_status": "approved",
        "source_class": "company_site",
        "evidence_span": f"Exact company site snapshot with {len(pages)} bounded pages and {len(social_links)} verified social links.",
        "metrics": {
            "bounded_pages_captured": len(pages),
            "verified_social_links": len(social_links),
            "structured_organisation_records": len(val.get("structured_organisations") or []),
            "extraction_state": val.get("extraction_state"),
        },
        "strategy": "company_site_activity",
    })

    # 2. Company-owned activity / news posts
    news_pages = [
        p for p in pages
        if NEWS_PATH.search(urlparse(str(p.get("url") or "")).path)
    ]
    if news_pages:
        news_pages.sort(key=lambda p: (-len([part for part in urlparse(str(p.get("url") or "")).path.split("/") if part]), str(p.get("url") or "")))
        selected_news = news_pages[0]
        news_url = str(selected_news.get("url") or "")
        news_hash = str(selected_news.get("content_sha256") or _digest(news_url))
        news_title = str(selected_news.get("title") or "Company news/activity page").strip()
        observations.append({
            "id": f"company-site-news-{org}-{_digest(org + news_url)[:16]}",
            "organisation_number": org,
            "platform": "company_site",
            "signal_type": "public_post",
            "source_url": news_url,
            "retrieved_at": retrieved_at,
            "content_sha256": news_hash,
            "exact_entity": True,
            "identity_proof": [{"type": "website_identity_gate", "score": identity.get("score")}],
            "acquisition_mode": "permitted_public_page",
            "rights_status": "approved",
            "source_class": "company_site",
            "evidence_span": news_title[:1200],
            "metrics": {
                "captured_news_pages": len(news_pages),
                "interpretation": "Company-owned activity; not independent sentiment.",
            },
            "strategy": "company_site_activity",
        })

    # 3. Verified social handles discovered on the company-controlled site
    for item in social_links:
        platform = str(item.get("platform") or "")
        url = str(item.get("url") or "")
        if not platform or not url:
            continue
        link_hash = _digest(f"{org}|{platform}|{url}")
        observations.append({
            "id": f"verified-handle-{org}-{link_hash[:16]}",
            "organisation_number": org,
            "platform": platform,
            "signal_type": "profile_handle",
            "source_url": url,
            "profile_url": url,
            "retrieved_at": retrieved_at,
            "content_sha256": link_hash,
            "exact_entity": True,
            "identity_proof": [
                {"type": "declared_on_exact_company_website", "final_url": source_url, "score": identity.get("score")}
            ],
            "acquisition_mode": "permitted_public_page",
            "rights_status": "approved",
            "source_class": "company_social",
            "evidence_span": f"Verified {platform} profile declared on exact company site: {url}",
            "strategy": "verified_handle_extraction",
        })

    return observations


def extract_registry_workforce_observation(profile: dict[str, Any]) -> dict[str, Any] | None:
    """Emit an official workforce snapshot if employee count is registered."""
    org = str(profile.get("organisation_number") or "")
    employees = profile.get("employees")
    if employees is None:
        reg_val = (profile.get("evidence", {}).get("registry_live", {}).get("value") or {})
        employees = reg_val.get("employees")
    if employees is None:
        return None

    reg_evidence = profile.get("evidence", {}).get("registry_live") or profile.get("evidence", {}).get("registry") or {}
    source_url = reg_evidence.get("source_url") or f"https://data.brreg.no/enhetsregisteret/api/enheter/{org}"
    retrieved_at = reg_evidence.get("retrieved_at") or utc_now()
    content_hash = reg_evidence.get("content_sha256") or _digest(f"{org}|employees|{employees}")

    return {
        "id": f"registry-workforce-{org}-{content_hash[:16]}",
        "organisation_number": org,
        "platform": "brreg",
        "signal_type": "workforce_snapshot",
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "content_sha256": content_hash,
        "exact_entity": True,
        "identity_proof": [
            {"type": "official_registry_record", "organisation_number": org}
        ],
        "acquisition_mode": "official_api",
        "rights_status": "approved",
        "source_class": "official_registry",
        "effective_at": str(profile.get("latest_submitted_accounts") or ""),
        "evidence_span": f"Official Brreg registration reports {employees} employees for entity {org}.",
        "metrics": {
            "workforce_value": employees,
            "measure": "registered_employees",
            "scope": "company_legal_entity",
        },
        "strategy": "registry_workforce_snapshot",
    }


def enrich_company_profile(
    profile: dict[str, Any],
    requested_modules: set[str] | list[str],
    *,
    fetcher: Callable[[str], FetchResult] = fetch_json,
    timeout: float = 15.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """
    Executes official Brreg and company website pipelines, synchronizes
    top-level entity attributes, and extracts approved external observations.
    """
    org = str(profile["organisation_number"])
    profile.setdefault("evidence", {})
    modules_set = set(requested_modules)
    
    official_modules = modules_set - {"registry", "accounting_obligation", "website"}
    official_records, official_metrics = fetch_official_modules(org, official_modules, fetcher=fetcher)
    profile["evidence"].update(official_records)

    # Sync official live updates to root profile fields
    live = official_records.get("registry_live")
    if live and live.get("status") == "available":
        live_val = live.get("value") or {}
        for src, dst in (
            ("name", "name"),
            ("legal_form", "legal_form"),
            ("employees", "employees"),
            ("website", "website"),
            ("latest_submitted_accounts", "latest_submitted_accounts"),
            ("bankrupt", "bankrupt"),
            ("liquidating", "liquidating"),
        ):
            if live_val.get(src) is not None:
                profile[dst] = live_val[src]

    # Handle company website crawl and identity gating
    web_metrics = {"requests": 0, "bytes": 0, "latencies_ms": []}
    if "website" in modules_set:
        site_url = profile.get("website")
        if site_url:
            web_record, web_metrics = fetch_website(site_url, timeout=timeout)
            gated = apply_website_identity_gate(profile, web_record)
            profile["evidence"]["website"] = gated["website"]
        else:
            profile["evidence"]["website"] = evidence(
                "website",
                "not_found",
                "registry_linked_company_website",
                "https://data.brreg.no/enhetsregisteret/api/enheter",
                note="No website declared in registry records",
            )

    # Compile performance & operational metrics
    operations = {
        "requests": len(official_metrics) + web_metrics.get("requests", 0),
        "bytes": sum(m.bytes_received for m in official_metrics) + web_metrics.get("bytes", 0),
        "latencies_ms": [m.elapsed_ms for m in official_metrics] + web_metrics.get("latencies_ms", []),
        "runtime_ms": sum(m.elapsed_ms for m in official_metrics) + sum(web_metrics.get("latencies_ms", [])),
        "third_party_cost_usd": 0.0,
    }
    profile["run_metrics"] = operations

    # Extract approved external-footprint observations
    observations = extract_site_observations(profile)
    workforce_obs = extract_registry_workforce_observation(profile)
    if workforce_obs:
        observations.append(workforce_obs)

    # Generate company summary synthesis (LLM if configured, deterministic fallback)
    try:
        from .llm import generate_company_summary_with_llm
        profile["company_summary"] = generate_company_summary_with_llm(profile)
    except Exception:
        pass

    return profile, observations, operations