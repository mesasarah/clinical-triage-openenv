"""
Deterministic graders for all three tasks.
Each grader returns a ClinicalReward with total in [0.0, 1.0].
All grading is reproducible — same input always yields same score.
"""
from __future__ import annotations

from typing import Any, Dict, List
from server.models import ClinicalReward, TriagePriority


# ──────────────────────────────────────────────
# TASK EASY — Triage Classification Grader
# ──────────────────────────────────────────────

# Priority adjacency penalty: distance between priority levels
_PRIORITY_ORDER = [
    TriagePriority.IMMEDIATE,
    TriagePriority.URGENT,
    TriagePriority.SEMI_URGENT,
    TriagePriority.NON_URGENT,
]

def _priority_distance(pred: str, truth: str) -> int:
    """Returns how many levels apart two priorities are (0 = exact match)."""
    try:
        p_idx = _PRIORITY_ORDER.index(TriagePriority(pred.upper()))
        t_idx = _PRIORITY_ORDER.index(TriagePriority(truth.upper()))
        return abs(p_idx - t_idx)
    except ValueError:
        return 4  # completely invalid answer


def grade_easy(action_content: str, ground_truth: str) -> ClinicalReward:
    """
    Score a single triage classification.
    - Exact match:              1.0
    - Off by 1 level:           0.5
    - Off by 2 levels:          0.2
    - Off by 3+ levels:         0.0
    - Lethal error (e.g. NON_URGENT when IMMEDIATE): 0.0 + penalty flag
    """
    pred = action_content.strip().upper()
    dist = _priority_distance(pred, ground_truth)

    score_map = {0: 1.0, 1: 0.5, 2: 0.2}
    total = score_map.get(dist, 0.0)

    # Extra safety check: calling IMMEDIATE or URGENT as NON_URGENT is lethal
    truth_level = _PRIORITY_ORDER.index(TriagePriority(ground_truth))
    lethal_error = False
    if truth_level <= 1 and dist >= 3:  # IMMEDIATE/URGENT → NON_URGENT
        total = 0.0
        lethal_error = True

    components = {
        "classification_score": total,
        "distance_penalty":     float(dist),
        "lethal_error":         float(lethal_error),
    }

    if total == 1.0:
        feedback = f"Correct. '{pred}' matches ground truth '{ground_truth}'."
    elif total > 0:
        feedback = f"Partial. Predicted '{pred}', expected '{ground_truth}' ({dist} level(s) off)."
    elif lethal_error:
        feedback = f"CRITICAL ERROR: Predicted '{pred}' for a '{ground_truth}' patient — potentially lethal under-triage."
    else:
        feedback = f"Incorrect. Predicted '{pred}', expected '{ground_truth}'."

    return ClinicalReward(total=total, components=components, feedback=feedback)


# ──────────────────────────────────────────────
# TASK MEDIUM — Diagnostic Test Ordering Grader
# ──────────────────────────────────────────────

def grade_medium(
    ordered_codes: List[str],
    required_tests: List[str],
    expected_tests: List[str],
    allowed_extras: List[str],
    forbidden_tests: List[str],
) -> ClinicalReward:
    """
    Multi-component scoring for test ordering:
      - Required coverage  (0.50): all safety-critical tests present
      - Expected coverage  (0.30): proportion of expected tests ordered
      - Over-ordering penalty (0.10): deduction for unnecessary/forbidden tests
      - Forbidden penalty  (0.10): hard deduction for forbidden tests ordered

    Final score is clamped to [0.0, 1.0].
    """
    ordered_set   = {c.upper().strip() for c in ordered_codes}
    required_set  = {c.upper() for c in required_tests}
    expected_set  = {c.upper() for c in expected_tests}
    allowed_set   = {c.upper() for c in allowed_extras}
    forbidden_set = {c.upper() for c in forbidden_tests}

    # 1. Required coverage (50%)
    missing_required = required_set - ordered_set
    required_score   = 1.0 - (len(missing_required) / len(required_set)) if required_set else 1.0
    required_contrib = required_score * 0.50

    # 2. Expected coverage (30%)
    expected_hit   = len(expected_set & ordered_set)
    expected_score = expected_hit / len(expected_set) if expected_set else 1.0
    expected_contrib = expected_score * 0.30

    # 3. Over-ordering penalty (10%) — tests not in expected or allowed
    acceptable      = expected_set | allowed_set | required_set
    unnecessary     = ordered_set - acceptable - forbidden_set
    over_penalty    = min(len(unnecessary) * 0.05, 0.10)
    over_contrib    = 0.10 - over_penalty

    # 4. Forbidden tests (10%) — hard penalty
    ordered_forbidden = ordered_set & forbidden_set
    forbidden_penalty = min(len(ordered_forbidden) * 0.05, 0.10)
    forbidden_contrib = 0.10 - forbidden_penalty

    total = max(0.0, min(1.0,
        required_contrib + expected_contrib + over_contrib + forbidden_contrib
    ))

    components = {
        "required_coverage":   round(required_contrib, 3),
        "expected_coverage":   round(expected_contrib, 3),
        "over_ordering":       round(over_contrib, 3),
        "forbidden_tests":     round(forbidden_contrib, 3),
        "missing_required":    list(missing_required),
        "ordered_forbidden":   list(ordered_forbidden),
        "unnecessary_ordered": list(unnecessary),
    }

    lines = [f"Score: {total:.2f}"]
    if missing_required:
        lines.append(f"MISSING safety-critical tests: {', '.join(sorted(missing_required))}")
    if ordered_forbidden:
        lines.append(f"INAPPROPRIATE tests ordered: {', '.join(sorted(ordered_forbidden))}")
    if unnecessary:
        lines.append(f"Unnecessary tests (minor penalty): {', '.join(sorted(unnecessary))}")
    hit_list = sorted(expected_set & ordered_set)
    if hit_list:
        lines.append(f"Correctly ordered: {', '.join(hit_list)}")

    return ClinicalReward(total=total, components=components, feedback=" | ".join(lines))


# ──────────────────────────────────────────────
# TASK HARD — Discharge Plan Grader
# ──────────────────────────────────────────────

def _keyword_hit_rate(text: str, keywords: List[str]) -> float:
    """Returns fraction of keywords found in text (case-insensitive)."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def _count_red_flags(text: str, red_flag_keywords: List[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in red_flag_keywords if kw.lower() in text_lower)


def grade_hard(
    discharge_text: str,
    required_keywords: Dict[str, List[str]],
    forbidden_keywords: List[str],
    min_red_flags: int,
) -> ClinicalReward:
    """
    Multi-criterion grader for discharge plan quality.

    Sections and weights:
      - Diagnosis accuracy       0.15
      - Medications coverage     0.20
      - Follow-up instructions   0.15
      - Red-flag warnings        0.25
      - Lifestyle/activity notes 0.15
      - No forbidden content     0.10
    """
    text = discharge_text.strip()

    # 1. Diagnosis (15%)
    diag_score    = _keyword_hit_rate(text, required_keywords.get("diagnosis", []))
    diag_contrib  = diag_score * 0.15

    # 2. Medications (20%)
    med_score     = _keyword_hit_rate(text, required_keywords.get("medications", []))
    med_contrib   = med_score * 0.20

    # 3. Follow-up (15%)
    fu_score      = _keyword_hit_rate(text, required_keywords.get("follow_up", []))
    fu_contrib    = fu_score * 0.15

    # 4. Red flags (25%) — count-based, must hit min_red_flags
    rf_keywords   = required_keywords.get("red_flags", [])
    rf_count      = _count_red_flags(text, rf_keywords)
    rf_score      = min(rf_count / max(min_red_flags, 1), 1.0)
    rf_contrib    = rf_score * 0.25

    # 5. Lifestyle / activity (15%) — use whichever key exists
    lifestyle_kws = required_keywords.get("lifestyle", required_keywords.get("activity", []))
    ls_score      = _keyword_hit_rate(text, lifestyle_kws)
    ls_contrib    = ls_score * 0.15

    # 6. Forbidden content penalty (10%)
    forbidden_hit = [kw for kw in forbidden_keywords if kw.lower() in text.lower()]
    forbidden_score = max(0.0, 1.0 - len(forbidden_hit) * 0.5)
    forbidden_contrib = forbidden_score * 0.10

    total = min(1.0, max(0.0,
        diag_contrib + med_contrib + fu_contrib + rf_contrib + ls_contrib + forbidden_contrib
    ))

    components = {
        "diagnosis":         round(diag_contrib,     3),
        "medications":       round(med_contrib,       3),
        "follow_up":         round(fu_contrib,        3),
        "red_flags":         round(rf_contrib,        3),
        "lifestyle":         round(ls_contrib,        3),
        "no_forbidden":      round(forbidden_contrib, 3),
        "red_flags_found":   rf_count,
        "red_flags_needed":  min_red_flags,
        "forbidden_found":   forbidden_hit,
    }

    lines = [f"Score: {total:.2f}"]
    lines.append(f"Diagnosis: {diag_score:.0%}  Meds: {med_score:.0%}  Follow-up: {fu_score:.0%}  Red-flags: {rf_count}/{min_red_flags}  Lifestyle: {ls_score:.0%}")
    if forbidden_hit:
        lines.append(f"WARNING — forbidden content detected: {', '.join(forbidden_hit)}")

    return ClinicalReward(total=total, components=components, feedback=" | ".join(lines))
