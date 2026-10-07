#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add src to the path so we can import our modules without weird hacks
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from norway_company_agent.batch import read_organisation_inputs, profiles_from_bulk, validate_envelopes
from norway_company_agent.evidence import utc_now
from norway_company_agent.enrichment import enrich_company_profile
from norway_company_agent.contract import build_unified_envelope
from norway_company_agent.llm import load_env, is_llm_available, get_llm_config

# Load .env file automatically on startup
load_env()

# Try to pull in the PDF workforce connector if it exists.
try:
    from scripts.run_annual_report_workforce_connector import collect as collect_workforce
except ImportError:
    collect_workforce = None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Safely write out the JSONL file so a crash doesn't corrupt half-written data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp_path.replace(path)


def read_previous_profiles(path: Path | str | None) -> dict[str, dict]:
    """Loads previous snapshot profiles or envelopes by organisation number for refresh diffing."""
    if not path:
        return {}
    prev_path = Path(path)
    if not prev_path.is_file():
        return {}
    results = {}
    try:
        for line in prev_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            org = str(item.get("organisation_number") or "")
            if not org:
                continue
            # Handle if previous input was an envelope or a profile
            profile = item.get("profile") if "profile" in item else item
            results[org] = profile
    except Exception as e:
        print(f"Warning: could not read previous snapshot from {path}: {e}")
    return results


def process_company(
    profile: dict,
    run_id: str,
    started_at: str,
    cache_dir: Path,
    modules: list[str],
    previous_profile: dict | None = None,
) -> dict:
    """
    Grabs official records and safe website data.
    Enriches with identity validation, optional PDF workforce and LLM synthesis.
    """
    # 1. Hit the Brreg APIs and do a strict-identity web crawl
    profile, observations, operations = enrich_company_profile(
        profile,
        requested_modules=modules,
        timeout=15.0,
    )

    # 2. Grab the official PDF employee counts if the tool is loaded
    if collect_workforce:
        try:
            # We keep OCR pages low so we don't timeout the whole run
            pdf_obs, pdf_status = collect_workforce(profile, cache_dir, ocr_pages=3, ocr_dpi=130)
            if pdf_obs:
                observations.append(pdf_obs)
        except Exception as e:
            # log it internally but don't blow up the company's run
            profile.setdefault("errors", []).append(f"PDF workforce error: {e}")

    # 3. Wrap it all in the dual-contract envelope (satisfies OUTPUT_CONTRACT.md and validate_envelopes)
    completed_at = utc_now()
    envelope = build_unified_envelope(
        profile=profile,
        run_id=run_id,
        modules=modules,
        started_at=started_at,
        completed_at=completed_at,
        previous_profile=previous_profile,
        extra_observations=observations,
    )

    return envelope


def main() -> None:
    parser = argparse.ArgumentParser(description="Official Signalpost run command")
    parser.add_argument("--organisations", required=True, help="Input list of org numbers (.json, .jsonl, .txt)")
    parser.add_argument("--bulk", required=True, help="Frozen Brreg bulk snapshot (.csv or .csv.gz)")
    parser.add_argument("--output", required=True, help="Final envelopes output path (.jsonl)")
    parser.add_argument("--previous", help="Optional previous snapshot path for refresh change detection")
    parser.add_argument("--report", help="Optional path for machine-readable run report JSON")
    parser.add_argument("--cache-dir", default=".cache", help="Where to stash PDF downloads")
    parser.add_argument("--workers", type=int, default=8, help="How many worker threads to run")
    args = parser.parse_args()

    run_start_time = time.monotonic()
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    started_at = utc_now()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Report LLM configuration status
    llm_cfg = get_llm_config()
    if llm_cfg:
        print(f"LLM integration: ACTIVE (provider: {llm_cfg['provider']}, model: {llm_cfg['model']})")
    else:
        print("LLM integration: INACTIVE (no API key in .env or environment; running in deterministic mode)")

    # Foundation and official modules
    modules = ["registry", "accounting_obligation", "registry_live", "financials", "roles", "group", "locations", "website"]

    print(f"Loading inputs from {args.organisations}...")
    org_inputs = read_organisation_inputs(args.organisations)
    org_numbers = [item["organisation_number"] for item in org_inputs]

    print(f"Extracting {len(org_numbers)} profiles from bulk registry...")
    profiles, registry_metadata = profiles_from_bulk(args.bulk, org_numbers)

    # Map previous profiles for refresh diffing if requested
    previous_profiles = read_previous_profiles(args.previous) if args.previous else {}
    if previous_profiles:
        print(f"Loaded {len(previous_profiles)} previous profiles for change detection.")

    # Map any extra annotations (like sample_slice) from the input to the profile
    annotations = {item["organisation_number"]: item for item in org_inputs}
    for profile in profiles:
        for key in ("evaluation_split", "sample_slice"):
            if key in annotations[profile["organisation_number"]]:
                profile[key] = annotations[profile["organisation_number"]][key]

    envelopes = []
    print(f"Processing companies with {args.workers} workers...")

    # Spin up thread pool to hit APIs concurrently
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_company,
                profile,
                run_id,
                started_at,
                cache_dir,
                modules,
                previous_profiles.get(profile["organisation_number"]),
            ): profile["organisation_number"]
            for profile in profiles
        }

        for i, future in enumerate(as_completed(futures), 1):
            org = futures[future]
            try:
                envelope = future.result()
                envelopes.append(envelope)
            except Exception as e:
                # Total failure fallback conforming to OUTPUT_CONTRACT.md
                print(f"Hard crash on {org}: {e}")
                envelopes.append({
                    "organisation_number": org,
                    "run": {
                        "run_id": run_id,
                        "terminal_status": "failed",
                        "started_at": started_at,
                        "completed_at": utc_now(),
                    },
                    "state": "submission_error",
                    "modules": {m: {"state": "submission_error"} for m in modules},
                    "claims": [],
                    "evidence": [],
                    "changes": [],
                    "errors": [str(e)],
                    "operations": {
                        "requests": 0,
                        "runtime_ms": 0,
                        "third_party_cost_usd": 0.0,
                    },
                })

            if i % 10 == 0 or i == len(profiles):
                print(f"Finished {i}/{len(profiles)}...")

    # Sort envelopes back to the original input order so the evaluator is happy
    order_map = {org: idx for idx, org in enumerate(org_numbers)}
    envelopes.sort(key=lambda e: order_map.get(e["organisation_number"], 9999))

    # Write final envelopes output
    output_path = Path(args.output)
    print(f"Writing {len(envelopes)} envelopes to {output_path}")
    write_jsonl(output_path, envelopes)

    # Validate output envelopes against contract
    validation = validate_envelopes(envelopes, len(org_numbers))
    total_elapsed_ms = int((time.monotonic() - run_start_time) * 1000)
    total_requests = sum(e.get("operations", {}).get("requests", 0) for e in envelopes)
    total_cost_usd = sum(e.get("operations", {}).get("third_party_cost_usd", 0.0) for e in envelopes)

    # Generate machine-readable run report
    report_path = Path(args.report) if args.report else output_path.parent / "run-report.json"
    report = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": utc_now(),
        "runtime_ms": total_elapsed_ms,
        "organisations_requested": len(org_numbers),
        "envelopes_emitted": len(envelopes),
        "validation": validation,
        "operations": {
            "total_requests": total_requests,
            "total_runtime_ms": total_elapsed_ms,
            "third_party_cost_usd": total_cost_usd,
            "avg_requests_per_company": round(total_requests / len(envelopes), 2) if envelopes else 0,
        },
        "registry": registry_metadata,
        "llm_configuration": {
            "active": is_llm_available(),
            "provider": llm_cfg["provider"] if llm_cfg else None,
            "model": llm_cfg["model"] if llm_cfg else None,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved run report to {report_path}")
    print(f"Validation passed: {validation['passed']} (zero silent drops: {validation['checks']['zero_silent_drops']})")
    print("Done.")


if __name__ == "__main__":
    main()