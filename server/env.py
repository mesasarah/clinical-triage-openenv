"""
ClinicalTriageEnv — core environment implementing the OpenEnv API.
Manages state, dispatches to task-specific graders, and builds observations.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

from server.models import (
    ClinicalAction,
    ClinicalObservation,
    ClinicalReward,
    PatientRecord,
    ResetResult,
    StateResult,
    StepResult,
    TaskID,
)
from data.fixtures import (
    DIAGNOSTIC_CATALOG,
    EASY_CASES,
    MEDIUM_CASES,
    HARD_CASES,
)
from graders.graders import grade_easy, grade_medium, grade_hard


# ──────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────

_EASY_PROMPT = """You are an emergency department triage nurse.

Read the patient record below and classify the patient into exactly ONE of:
  IMMEDIATE   — life-threatening, act within seconds/minutes
  URGENT      — needs assessment within 30 minutes
  SEMI_URGENT — needs assessment within 2 hours
  NON_URGENT  — routine care, no immediate risk

Reply with ONLY the classification word. Nothing else.

Patient ID: {patient_id}
Age / Sex: {age} / {sex}
Chief Complaint: {complaint}
History: {history}
Vitals:
  Heart Rate:        {hr} bpm
  Blood Pressure:    {bp}
  Respiratory Rate:  {rr} breaths/min
  Temperature:       {temp}°C
  SpO2:              {spo2}%
  GCS:               {gcs}/15
Allergies: {allergies}
Current Medications: {meds}
"""

_MEDIUM_PROMPT = """You are an emergency physician completing an initial workup.

The patient has been triaged as {triage}. Review the clinical picture and order
the appropriate diagnostic tests from the catalog below.

Reply with a comma-separated list of test CODES ONLY. No spaces around commas.
Example: CBC,CMP,ECG,TROPI

Patient ID: {patient_id}
Age / Sex: {age} / {sex}
Chief Complaint: {complaint}
History: {history}
Vitals:
  HR: {hr} bpm  |  BP: {bp}  |  RR: {rr}  |  Temp: {temp}°C  |  SpO2: {spo2}%  |  GCS: {gcs}

AVAILABLE TEST CATALOG:
{catalog}

Reply with test codes only.
"""

_HARD_PROMPT = """You are an emergency physician preparing a discharge summary.

The patient is medically stable and ready for discharge. Write a complete,
clinically safe discharge plan covering ALL sections below.

Patient ID: {patient_id}  |  Age: {age}  |  Sex: {sex}
Diagnosis: {diagnosis}
Clinical Summary: {history}
Allergies: {allergies}
Current Medications on Admission: {meds}

Your discharge plan MUST include these clearly labelled sections:

1. DIAGNOSIS
   State the confirmed discharge diagnosis.

2. MEDICATIONS
   List all medications with dose, frequency, and duration.
   Flag any medications the patient must NOT take (allergy, drug interaction).

3. FOLLOW-UP
   Specify which specialty/GP, timeframe, and any tests to repeat.

4. RED FLAGS — RETURN TO ED IF:
   List at least {min_red_flags} specific warning signs that should prompt immediate return.

5. ACTIVITY & DIET
   Specify restrictions and when normal activity can resume.

Write clearly. A patient and their family must understand this document.
"""


def _fmt_patient(p: PatientRecord) -> Dict[str, str]:
    return {
        "patient_id": p.patient_id,
        "age":        str(p.age),
        "sex":        p.sex,
        "complaint":  p.chief_complaint,
        "history":    p.history,
        "hr":         str(p.vitals.heart_rate),
        "bp":         p.vitals.blood_pressure,
        "rr":         str(p.vitals.respiratory_rate),
        "temp":       str(p.vitals.temperature_c),
        "spo2":       str(p.vitals.spo2_percent),
        "gcs":        str(p.vitals.gcs),
        "allergies":  ", ".join(p.allergies) if p.allergies else "None",
        "meds":       ", ".join(p.current_meds) if p.current_meds else "None",
    }


# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────

class ClinicalTriageEnv:
    """
    OpenEnv-compliant environment for clinical triage tasks.
    One instance per episode; call reset() to begin a new episode.
    """

    MAX_STEPS: Dict[str, int] = {
        TaskID.EASY:   5,
        TaskID.MEDIUM: 10,
        TaskID.HARD:   15,
    }

    def __init__(self) -> None:
        self._task_id:     str                        = TaskID.EASY
        self._step:        int                        = 0
        self._done:        bool                       = False
        self._total_reward: float                     = 0.0
        self._history:     List[Dict[str, Any]]       = []
        self._case:        Optional[Dict[str, Any]]   = None
        self._patient:     Optional[PatientRecord]    = None
        self._last_obs:    Optional[ClinicalObservation] = None

    # ── OpenEnv API ──────────────────────────────

    def reset(self, task_id: str = TaskID.EASY) -> ResetResult:
        """Start a new episode for the given task."""
        self._task_id     = task_id
        self._step        = 0
        self._done        = False
        self._total_reward = 0.0
        self._history     = []

        # Pick a deterministic case (index 0 for reproducibility)
        if task_id == TaskID.EASY:
            self._case    = EASY_CASES[0]
        elif task_id == TaskID.MEDIUM:
            self._case    = MEDIUM_CASES[0]
        else:
            self._case    = HARD_CASES[0]

        self._patient = self._case["patient"]
        obs = self._build_observation()
        self._last_obs = obs
        return ResetResult(observation=obs)

    def step(self, action: ClinicalAction) -> StepResult:
        """Process one agent action and return (observation, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step += 1
        reward_obj = self._grade(action.content)

        self._total_reward += reward_obj.total
        self._history.append({
            "step":     self._step,
            "action":   action.content[:200],
            "reward":   reward_obj.total,
            "feedback": reward_obj.feedback,
        })

        # Episode ends on first valid response or when max steps reached
        max_steps = self.MAX_STEPS.get(self._task_id, 10)
        self._done = (reward_obj.total > 0.0) or (self._step >= max_steps)

        obs = self._build_observation()
        self._last_obs = obs

        return StepResult(
            observation=obs,
            reward=reward_obj.total,
            done=self._done,
            info={
                "feedback":    reward_obj.feedback,
                "components":  reward_obj.components,
                "step":        self._step,
                "total_reward": self._total_reward,
            },
        )

    def state(self) -> StateResult:
        """Return the current episode state (non-destructive)."""
        return StateResult(
            task_id      = self._task_id,
            step         = self._step,
            done         = self._done,
            total_reward = self._total_reward,
            patient_id   = self._patient.patient_id if self._patient else "",
            history      = copy.deepcopy(self._history),
        )

    # ── Internal helpers ─────────────────────────

    def _build_observation(self) -> ClinicalObservation:
        p    = self._patient
        case = self._case

        if self._task_id == TaskID.EASY:
            prompt = _EASY_PROMPT.format(**_fmt_patient(p))
            return ClinicalObservation(
                task_id  = self._task_id,
                step     = self._step,
                patient  = p,
                prompt   = prompt,
            )

        elif self._task_id == TaskID.MEDIUM:
            catalog_lines = "\n".join(
                f"  {t.code:<10} {t.name}  —  {t.description}"
                for t in DIAGNOSTIC_CATALOG
            )
            prompt = _MEDIUM_PROMPT.format(
                triage=case["triage"],
                catalog=catalog_lines,
                **_fmt_patient(p),
            )
            return ClinicalObservation(
                task_id        = self._task_id,
                step           = self._step,
                patient        = p,
                available_tests= DIAGNOSTIC_CATALOG,
                ordered_tests  = [],
                prompt         = prompt,
            )

        else:  # HARD
            prompt = _HARD_PROMPT.format(
                diagnosis   = case["diagnosis"],
                min_red_flags = case["min_red_flags"],
                **_fmt_patient(p),
            )
            return ClinicalObservation(
                task_id   = self._task_id,
                step      = self._step,
                patient   = p,
                diagnosis = case["diagnosis"],
                prompt    = prompt,
            )

    def _grade(self, content: str) -> ClinicalReward:
        case = self._case

        if self._task_id == TaskID.EASY:
            return grade_easy(content, case["ground_truth"])

        elif self._task_id == TaskID.MEDIUM:
            codes = [c.strip().upper() for c in content.split(",") if c.strip()]
            return grade_medium(
                ordered_codes   = codes,
                required_tests  = case["required_tests"],
                expected_tests  = case["expected_tests"],
                allowed_extras  = case["allowed_extras"],
                forbidden_tests = case["forbidden_tests"],
            )

        else:  # HARD
            return grade_hard(
                discharge_text    = content,
                required_keywords = case["required_keywords"],
                forbidden_keywords= case["forbidden_keywords"],
                min_red_flags     = case["min_red_flags"],
            )
