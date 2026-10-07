#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# point python to our src directory so we can import internal modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from norway_company_agent.external_footprint import (
    PUBLISHABLE_ACQUISITION_MODES,
    publishable_observation,
    validate_observation,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Helper to parse a jsonl file line by line while skipping blanks."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ratio(numerator: int, denominator: int) -> float:
    """Safe division helper to avoid ZeroDivisionError."""
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen external-footprint observations and exact-entity labels.")
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--labels", required=True, help="JSONL: id, exact_entity, metric_correct, sentiment_correct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-audit", type=int, default=100)
    args = parser.parse_args()

    profiles = read_jsonl(Path(args.profiles))
    observations = read_jsonl(Path(args.observations))
    label_rows = read_jsonl(Path(args.labels))
    
    # map label entries by id so we can look them up instantly
    labels = {str(item["id"]): item for item in label_rows}
    audit_size_gate = len(labels) >= args.minimum_audit
    
    if len(labels) != len(label_rows):
        raise ValueError("Label IDs must be unique across the audit set")

    # isolate the observations that were audited by humans/judges
    audited = [item for item in observations if str(item.get("id")) in labels]
    published = [item for item in audited if publishable_observation(item)]
    
    # count any hallucinations or errors against the audit labels
    wrong_entity = sum(not labels[str(item["id"])].get("exact_entity", False) for item in published)
    wrong_metric = sum(not labels[str(item["id"])].get("metric_correct", False) for item in published)
    unsupported = sum(bool(validate_observation(item)) for item in published)
    
    # check how sentiment models performed if any sentiment items were audited
    sentiment_audited = [item for item in published if item.get("sentiment_label") is not None]
    sentiment_correct = sum(labels[str(item["id"])].get("sentiment_correct", False) for item in sentiment_audited)

    all_orgs = {str(item["organisation_number"]) for item in profiles}
    accepted_all = [item for item in observations if publishable_observation(item)]
    
    org_platforms: dict[str, set[str]] = defaultdict(set)
    org_signals: dict[str, set[str]] = defaultdict(set)
    for item in accepted_all:
        org = str(item["organisation_number"])
        if org not in all_orgs:
            continue
        org_platforms[org].add(str(item["platform"]))
        org_signals[org].add(str(item["signal_type"]))

    # measure multi-platform coverage across the entire test corpus
    coverage = {
        "any_external": ratio(sum(bool(org_platforms[org]) for org in all_orgs), len(all_orgs)),
        "two_platforms": ratio(sum(len(org_platforms[org]) >= 2 for org in all_orgs), len(all_orgs)),
        "workforce_jobs": ratio(sum(bool(org_signals[org] & {"job_posting", "workforce_snapshot"}) for org in all_orgs), len(all_orgs)),
        "ratings_reviews": ratio(sum(bool(org_signals[org] & {"review", "review_summary", "place_summary"}) for org in all_orgs), len(all_orgs)),
        "buzz_engagement": ratio(sum(bool(org_signals[org] & {"public_post", "public_mention", "profile_metrics"}) for org in all_orgs), len(all_orgs)),
        "sentiment": ratio(sum(any(item.get("sentiment_label") for item in accepted_all if str(item.get("organisation_number")) == org) for org in all_orgs), len(all_orgs)),
    }
    
    # calculate freshness coverage (observations retrieved recently)
    now = datetime.now(timezone.utc)
    fresh_orgs = set()
    for item in accepted_all:
        retrieved_raw = item.get("retrieved_at")
        if not retrieved_raw:
            continue
        try:
            retrieved_dt = datetime.fromisoformat(str(retrieved_raw).replace("Z", "+00:00"))
            # consider items retrieved within the last 90 days as fresh
            if 0 <= (now - retrieved_dt).total_seconds() <= 90 * 86400:
                fresh_orgs.add(str(item.get("organisation_number")))
        except ValueError:
            pass
    fresh_coverage = ratio(len(fresh_orgs & all_orgs), len(all_orgs))

    acquisition_modes = Counter(str(item.get("acquisition_mode")) for item in observations)
    entity_precision = ratio(len(published) - wrong_entity, len(published))
    metric_precision = ratio(len(published) - wrong_metric, len(published))
    sentiment_accuracy = ratio(sentiment_correct, len(sentiment_audited)) if sentiment_audited else None

    # core qualification gate from original rubric
    qualification = bool(
        audit_size_gate
        and published
        and wrong_entity == 0
        and unsupported == 0
        and entity_precision >= 0.995
        and metric_precision >= 0.98
    )

    # verify all published items use approved acquisition modes and rights
    unapproved_modes = sum(
        item.get("acquisition_mode") not in PUBLISHABLE_ACQUISITION_MODES or item.get("rights_status") != "approved"
        for item in published
    )
    connector_policy_passed = bool(
        published
        and unsupported == 0
        and unapproved_modes == 0
    )

    report = {
        "scorer": "signalpost_external_footprint_eval_v1",
        "claim_boundary": "Held-out observation audit plus full-corpus coverage; it does not validate an unlabelled connector.",
        "profiles": len(profiles),
        "observations": len(observations),
        "audited_observations": len(audited),
        "published_audited": len(published),
        "wrong_entity_publications": wrong_entity,
        "unsupported_publications": unsupported,
        "entity_precision": entity_precision,
        "metric_precision": metric_precision,
        "sentiment_audited": len(sentiment_audited),
        "sentiment_accuracy": sentiment_accuracy,
        "coverage": coverage,
        "fresh_coverage": fresh_coverage,
        "platform_counts": dict(Counter(str(item.get("platform")) for item in accepted_all)),
        "acquisition_modes": dict(acquisition_modes),
        "minimum_audit": args.minimum_audit,
        "audit_size_gate": audit_size_gate,
        "qualification_passed": qualification,
        "connector_policy_passed": connector_policy_passed,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()