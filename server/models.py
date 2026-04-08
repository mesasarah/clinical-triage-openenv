"""
Typed Pydantic models for the Clinical Triage Navigator OpenEnv environment.
All models are fully typed and serialisable.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class TriagePriority(str, Enum):
    IMMEDIATE   = "IMMEDIATE"    # life-threatening, act now
    URGENT      = "URGENT"       # within 30 minutes
    SEMI_URGENT = "SEMI_URGENT"  # within 2 hours
    NON_URGENT  = "NON_URGENT"   # routine


class TaskID(str, Enum):
    EASY   = "task_easy"
    MEDIUM = "task_medium"
    HARD   = "task_hard"


# ──────────────────────────────────────────────
# Patient & Clinical Data
# ──────────────────────────────────────────────

class Vitals(BaseModel):
    heart_rate:       int    = Field(..., description="Beats per minute")
    blood_pressure:   str    = Field(..., description="Systolic/Diastolic e.g. '120/80'")
    respiratory_rate: int    = Field(..., description="Breaths per minute")
    temperature_c:    float  = Field(..., description="Body temperature in Celsius")
    spo2_percent:     int    = Field(..., description="Oxygen saturation %")
    gcs:              int    = Field(..., description="Glasgow Coma Scale 3-15")


class PatientRecord(BaseModel):
    patient_id:       str
    age:              int
    sex:              str
    chief_complaint:  str
    history:          str
    vitals:           Vitals
    allergies:        List[str] = Field(default_factory=list)
    current_meds:     List[str] = Field(default_factory=list)
    metadata:         Dict[str, Any] = Field(default_factory=dict)


class DiagnosticTest(BaseModel):
    code:        str
    name:        str
    category:    str
    description: str


class DischargeInstruction(BaseModel):
    diagnosis:       str
    medications:     List[str]
    follow_up:       str
    red_flags:       List[str]
    activity_notes:  str
    diet_notes:      str


# ──────────────────────────────────────────────
# OpenEnv Core Models
# ──────────────────────────────────────────────

class ClinicalObservation(BaseModel):
    """Observation returned to the agent after reset() or step()."""
    task_id:           str
    step:              int
    patient:           PatientRecord
    available_tests:   Optional[List[DiagnosticTest]] = None
    ordered_tests:     Optional[List[str]] = None       # test codes already ordered
    test_results:      Optional[Dict[str, str]] = None  # code -> result string
    diagnosis:         Optional[str] = None             # provided in hard task
    prompt:            str                              # natural-language instruction
    episode_info:      Dict[str, Any] = Field(default_factory=dict)


class ClinicalAction(BaseModel):
    """Action submitted by the agent via step()."""
    content: str = Field(
        ...,
        description=(
            "For task_easy: one of IMMEDIATE | URGENT | SEMI_URGENT | NON_URGENT. "
            "For task_medium: comma-separated test codes e.g. 'CBC,CMP,ECG'. "
            "For task_hard: structured discharge plan as described in the prompt."
        )
    )


class ClinicalReward(BaseModel):
    """Detailed reward breakdown."""
    total:      float = Field(..., ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    feedback:   str = ""


class StepResult(BaseModel):
    observation: ClinicalObservation
    reward:      float
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: ClinicalObservation


class StateResult(BaseModel):
    task_id:      str
    step:         int
    done:         bool
    total_reward: float
    patient_id:   str
    history:      List[Dict[str, Any]] = Field(default_factory=list)
