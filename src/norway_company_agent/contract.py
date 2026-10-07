from __future__ import annotations

import hashlib
from typing import Any, Iterable
from .evidence import utc_now
from .refresh import diff_profile


TERMINAL_STATES = {
    "complete",
    "not_applicable",
    "not_found",
    "blocked_policy",
    "blocked_robots",
    "source_error",
    "budget_exhausted",
    "submission_error",
}


def evidence_terminal_state(record: dict[str, Any] | None) -> str:
    if not record:
        return "submission_error"
    status = record.get("status")
    if status == "available":
        return "complete"
    if status == "not_applicable":
        return "not_applicable"
    if status == "not_found":
        return "not_found"
    if status == "blocked":
        note = str(record.get("note") or "").casefold()
        return "blocked_robots" if "robot" in note else "blocked_policy"
    if status == "source_error":
        return "source_error"
    return "submission_error"


def _make_evidence_id(source_url: str, field: str, seed: str) -> str:
    digest = hashlib.sha256(f"{source_url}|{field}|{seed}".encode("utf-8")).hexdigest()
    return f"ev-{digest[:16]}"


def build_unified_envelope(
    profile: dict[str, Any],
    *,
    run_id: str,
    modules: Iterable[str],
    started_at: str,
    completed_at: str,
    previous_profile: dict[str, Any] | None = None,
    extra_observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    org = str(profile.get("organisation_number") or "")
    records = profile.get("evidence", {})
    requested_modules = list(modules)

    module_states: dict[str, dict[str, Any]] = {}
    for module in requested_modules:
        record = records.get(module)
        module_states[module] = {
            "state": evidence_terminal_state(record),
            "retry_count": int((record or {}).get("retry_count") or 0),
            "final_timestamp": (record or {}).get("retrieved_at") or completed_at,
        }

    has_submission_error = any(
        m["state"] == "submission_error" for m in module_states.values()
    )
    entity_state = "submission_error" if has_submission_error else "complete"

    claims: list[dict[str, Any]] = []
    evidence_list: list[dict[str, Any]] = []
    seen_evidence_ids: set[str] = set()

    def register_claim(
        field: str,
        value: Any,
        evidence_record: dict[str, Any] | None,
        source_class: str,
        confidence: float = 1.0,
        span_override: str | None = None,
        forced_availability: str | None = None,
    ) -> None:
        rec = evidence_record or {}
        status = rec.get("status")
        
        if forced_availability:
            availability = forced_availability
        elif status == "available" and value is not None:
            availability = "available"
        elif status == "blocked":
            availability = "blocked"
        elif status == "not_applicable":
            availability = "not_applicable"
        elif status in {"not_found", "available"} and value is None:
            availability = "not_available"
        else:
            availability = "failed"

        source_url = str(rec.get("source_url") or f"https://data.brreg.no/enhetsregisteret/api/enheter/{org}")
        content_hash = str(rec.get("content_sha256") or hashlib.sha256(str(value or "").encode()).hexdigest())
        ev_id = _make_evidence_id(source_url, field, org)

        claims.append({
            "field": field,
            "value": value,
            "availability": availability,
            "confidence": confidence if availability == "available" else 0.0,
            "evidence_ids": [ev_id],
        })

        if ev_id not in seen_evidence_ids:
            evidence_list.append({
                "id": ev_id,
                "source_url": source_url,
                "source_class": source_class,
                "retrieved_at": rec.get("retrieved_at") or completed_at,
                "content_sha256": content_hash,
                "claim_span": span_override or f"{field}: {str(value)[:400]}",
            })
            seen_evidence_ids.add(ev_id)

    # 1. Identity & Foundation Claims
    reg = records.get("registry_live") or records.get("registry") or {}
    reg_val = reg.get("value") or profile
    register_claim("company_name", reg_val.get("name") or profile.get("name"), reg, "official_registry")
    register_claim("legal_form", reg_val.get("legal_form") or profile.get("legal_form"), reg, "official_registry")
    register_claim("municipality", reg_val.get("municipality") or profile.get("municipality"), reg, "official_registry")
    register_claim("industry_code", reg_val.get("industry_code") or profile.get("industry_code"), reg, "official_registry")
    register_claim("registered_employees", reg_val.get("employees") if reg_val.get("employees") is not None else profile.get("employees"), reg, "official_registry")

    # 2. Leadership Claims (Roles)
    roles_rec = records.get("roles") or {}
    roles_list = (roles_rec.get("value") or {}).get("roles") or []
    active_roles = [r for r in roles_list if not r.get("inactive")]
    
    ceo = next((r.get("name") for r in active_roles if r.get("role_code") == "DAGL"), None)
    chair = next((r.get("name") for r in active_roles if r.get("role_code") == "LEDE"), None)
    board_members = [r.get("name") for r in active_roles if r.get("role_code") in {"MEDL", "LEDE"} and r.get("name")]
    
    register_claim("chief_executive", ceo, roles_rec, "official_roles")
    register_claim("board_chair", chair, roles_rec, "official_roles")
    register_claim("board_members", board_members if board_members else None, roles_rec, "official_roles")

    # 3. Workplaces / Subunits (Locations)
    loc_rec = records.get("locations") or {}
    locations_list = (loc_rec.get("value") or {}).get("locations") or []
    subunits = [
        {"name": l.get("name"), "address": l.get("address"), "org": l.get("organisation_number")}
        for l in locations_list
    ]
    register_claim("operating_subunits", subunits if subunits else None, loc_rec, "official_subunits")

    # 4. Financial Filings (Latest Year and Normalized History)
    fin_rec = records.get("financials") or {}
    fin_records = (fin_rec.get("value") or {}).get("records") or []
    if fin_records:
        latest = fin_records[0]
        period = latest.get("period") or {}
        register_claim("latest_accounts_period", f"{period.get('fraDato')} to {period.get('tilDato')}", fin_rec, "official_annual_accounts")
        register_claim("annual_revenue", latest.get("revenue"), fin_rec, "official_annual_accounts")
        register_claim("operating_result", latest.get("operating_result"), fin_rec, "official_annual_accounts")
        register_claim("annual_result", latest.get("annual_result"), fin_rec, "official_annual_accounts")
        register_claim("total_assets", latest.get("assets"), fin_rec, "official_annual_accounts")
        register_claim("total_debt", latest.get("debt"), fin_rec, "official_annual_accounts")
        
        # 3-Year Historical Accounts
        history = [
            {
                "year": (r.get("period") or {}).get("tilDato", "")[:4],
                "revenue": r.get("revenue"),
                "operating_result": r.get("operating_result"),
                "annual_result": r.get("annual_result"),
            }
            for r in fin_records[:3]
        ]
        register_claim("financial_history", history, fin_rec, "official_annual_accounts")
    else:
        for f in ("latest_accounts_period", "annual_revenue", "operating_result", "annual_result", "total_assets", "total_debt", "financial_history"):
            register_claim(f, None, fin_rec, "official_annual_accounts", forced_availability="not_available")

    # 5. Website and Public Presence
    web_rec = records.get("website") or {}
    web_val = web_rec.get("value") or {}
    identity = web_val.get("identity_assessment") or {}
    
    if web_rec.get("status") == "available" and identity.get("publishable"):
        final_url = web_val.get("final_url") or web_rec.get("source_url")
        register_claim("official_website", final_url, web_rec, "company_owned", confidence=0.99)
        register_claim("website_description", web_val.get("description") or None, web_rec, "company_owned", confidence=0.95)
        
        social_links = web_val.get("social_links") or []
        register_claim("social_channels", [s.get("url") for s in social_links] if social_links else None, web_rec, "company_owned")
    else:
        avail = "blocked" if web_rec.get("status") == "blocked" else "not_available"
        register_claim("official_website", None, web_rec, "company_owned", forced_availability=avail)
        register_claim("website_description", None, web_rec, "company_owned", forced_availability=avail)
        register_claim("social_channels", None, web_rec, "company_owned", forced_availability=avail)

    # 5b. Group structure
    grp_rec = records.get("group") or {}
    grp_val = grp_rec.get("value")
    if grp_val:
        register_claim("group_structure", grp_val, grp_rec, "official_group_structure")

    # 5c. Company Synthesis / Research Summary (LLM or deterministic)
    if profile.get("company_summary"):
        summary_text = str(profile["company_summary"])
        source_url = f"https://data.brreg.no/enhetsregisteret/api/enheter/{org}"
        summary_hash = hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
        ev_id = _make_evidence_id(source_url, "company_summary", org)
        claims.append({
            "field": "company_summary",
            "value": summary_text,
            "availability": "available",
            "confidence": 0.95,
            "evidence_ids": [ev_id],
        })
        if ev_id not in seen_evidence_ids:
            evidence_list.append({
                "id": ev_id,
                "source_url": source_url,
                "source_class": "research_synthesis",
                "retrieved_at": completed_at,
                "content_sha256": summary_hash,
                "claim_span": summary_text[:400],
            })
            seen_evidence_ids.add(ev_id)

    # 6. Approved Extra Observations
    for obs in extra_observations or []:
        if str(obs.get("organisation_number")) != org:
            continue
        sig_type = obs.get("signal_type")
        obs_id = obs.get("id") or _make_evidence_id(obs.get("source_url", ""), sig_type, org)
        claims.append({
            "field": f"external_{sig_type}",
            "value": obs.get("metrics") or obs.get("evidence_span"),
            "availability": "available",
            "confidence": 0.99 if obs.get("exact_entity") else 0.8,
            "evidence_ids": [obs_id],
        })
        if obs_id not in seen_evidence_ids:
            evidence_list.append({
                "id": obs_id,
                "source_url": obs.get("source_url"),
                "source_class": obs.get("source_class") or "public_evidence",
                "retrieved_at": obs.get("retrieved_at") or completed_at,
                "content_sha256": obs.get("content_sha256") or hashlib.sha256(b"").hexdigest(),
                "claim_span": obs.get("evidence_span") or str(obs.get("metrics"))[:400],
            })
            seen_evidence_ids.add(obs_id)

    changes = []
    if previous_profile:
        changes = diff_profile(previous_profile, profile)

    raw_ops = profile.get("run_metrics") or {}
    operations = {
        "requests": int(raw_ops.get("requests", sum(1 for m in module_states.values() if m["state"] != "not_applicable"))),
        "runtime_ms": int(raw_ops.get("runtime_ms", 0)),
        "third_party_cost_usd": float(raw_ops.get("third_party_cost_usd", 0.0)),
    }

    return {
        "organisation_number": org,
        "run": {
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "terminal_status": "completed" if entity_state == "complete" else "failed",
        },
        "claims": claims,
        "evidence": evidence_list,
        "changes": changes,
        "errors": profile.get("errors", []),
        "operations": operations,
        "state": entity_state,
        "started_at": started_at,
        "completed_at": completed_at,
        "modules": module_states,
        "profile": profile,
    }