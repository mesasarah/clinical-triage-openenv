"""
Test suite for Clinical Triage Navigator.
Tests graders deterministically and validates env API contracts.
Run: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from graders.graders import grade_easy, grade_medium, grade_hard
from server.env import ClinicalTriageEnv
from server.models import ClinicalAction, TaskID


# ──────────────────────────────────────────────
# Easy grader tests
# ──────────────────────────────────────────────

class TestEasyGrader:

    def test_exact_match_immediate(self):
        r = grade_easy("IMMEDIATE", "IMMEDIATE")
        assert r.total == 1.0

    def test_exact_match_non_urgent(self):
        r = grade_easy("NON_URGENT", "NON_URGENT")
        assert r.total == 1.0

    def test_off_by_one(self):
        r = grade_easy("URGENT", "IMMEDIATE")
        assert r.total == 0.5

    def test_off_by_two(self):
        r = grade_easy("SEMI_URGENT", "IMMEDIATE")
        assert r.total == 0.2

    def test_lethal_error(self):
        r = grade_easy("NON_URGENT", "IMMEDIATE")
        assert r.total == 0.0
        assert r.components["lethal_error"] == 1.0

    def test_invalid_input(self):
        r = grade_easy("MAYBE_URGENT", "URGENT")
        assert r.total == 0.0

    def test_case_insensitive(self):
        r = grade_easy("immediate", "IMMEDIATE")
        assert r.total == 1.0

    def test_reward_in_range(self):
        for pred in ["IMMEDIATE", "URGENT", "SEMI_URGENT", "NON_URGENT"]:
            for truth in ["IMMEDIATE", "URGENT", "SEMI_URGENT", "NON_URGENT"]:
                r = grade_easy(pred, truth)
                assert 0.0 <= r.total <= 1.0


# ──────────────────────────────────────────────
# Medium grader tests
# ──────────────────────────────────────────────

class TestMediumGrader:

    def test_perfect_order(self):
        r = grade_medium(
            ordered_codes   = ["ECG", "TROPI", "CXR", "CMP", "CBC", "COAG"],
            required_tests  = ["ECG", "TROPI"],
            expected_tests  = ["ECG", "TROPI", "CXR", "CMP", "CBC", "COAG"],
            allowed_extras  = ["ABG", "BNP"],
            forbidden_tests = ["BHCG", "TSH"],
        )
        assert r.total >= 0.9

    def test_missing_required_test(self):
        r = grade_medium(
            ordered_codes   = ["CXR", "CMP", "CBC"],  # Missing ECG and TROPI
            required_tests  = ["ECG", "TROPI"],
            expected_tests  = ["ECG", "TROPI", "CXR", "CMP", "CBC", "COAG"],
            allowed_extras  = [],
            forbidden_tests = [],
        )
        assert r.total <= 0.5
        assert "ECG" in r.components["missing_required"]
        assert "TROPI" in r.components["missing_required"]

    def test_forbidden_test_penalty(self):
        r = grade_medium(
            ordered_codes   = ["ECG", "TROPI", "BHCG"],  # BHCG forbidden
            required_tests  = ["ECG", "TROPI"],
            expected_tests  = ["ECG", "TROPI"],
            allowed_extras  = [],
            forbidden_tests = ["BHCG"],
        )
        assert "BHCG" in r.components["ordered_forbidden"]
        assert r.components["forbidden_tests"] < 0.10

    def test_reward_always_in_range(self):
        for codes in [[], ["ECG"], ["ECG", "TROPI"], ["BHCG", "TSH"]]:
            r = grade_medium(
                ordered_codes   = codes,
                required_tests  = ["ECG", "TROPI"],
                expected_tests  = ["ECG", "TROPI", "CXR"],
                allowed_extras  = [],
                forbidden_tests = ["BHCG", "TSH"],
            )
            assert 0.0 <= r.total <= 1.0, f"Out of range for {codes}: {r.total}"


# ──────────────────────────────────────────────
# Hard grader tests
# ──────────────────────────────────────────────

class TestHardGrader:

    REQUIRED = {
        "diagnosis":   ["pancreatitis"],
        "medications": ["paracetamol", "ibuprofen"],
        "follow_up":   ["gastroenterology", "gp"],
        "red_flags":   ["fever", "worsening pain", "unable to eat", "jaundice", "vomiting"],
        "lifestyle":   ["alcohol", "abstain"],
    }

    def test_perfect_plan(self):
        plan = """
        DIAGNOSIS: Acute mild pancreatitis.
        MEDICATIONS: Paracetamol 1g QDS, Ibuprofen 400mg TDS, Ondansetron 4mg PRN, Omeprazole 20mg OD.
        FOLLOW-UP: Gastroenterology clinic in 2 weeks. Also see your GP in 1 week.
        RED FLAGS — RETURN TO ED IF: fever above 38, worsening pain, unable to eat or drink,
        jaundice, persistent vomiting, or any new symptoms.
        ACTIVITY: Abstain from alcohol entirely. Light activity only for 2 weeks.
        """
        r = grade_hard(plan, self.REQUIRED, ["morphine", "opioid"], 3)
        assert r.total >= 0.7

    def test_empty_plan(self):
        r = grade_hard("", self.REQUIRED, ["morphine"], 3)
        assert r.total == 0.0

    def test_forbidden_keyword_penalty(self):
        plan = "DIAGNOSIS: pancreatitis. MEDICATIONS: morphine 10mg IV. Abstain from alcohol."
        r = grade_hard(plan, self.REQUIRED, ["morphine"], 3)
        assert r.components["no_forbidden"] < 0.10

    def test_reward_always_in_range(self):
        for plan in ["", "pancreatitis", "fever pain vomiting abstain alcohol gastroenterology paracetamol ibuprofen jaundice worsening pain unable to eat gp"]:
            r = grade_hard(plan, self.REQUIRED, ["morphine"], 3)
            assert 0.0 <= r.total <= 1.0


# ──────────────────────────────────────────────
# Environment API contract tests
# ──────────────────────────────────────────────

class TestEnvContract:

    def test_reset_returns_observation(self):
        env = ClinicalTriageEnv()
        result = env.reset(TaskID.EASY)
        assert result.observation is not None
        assert result.observation.patient is not None
        assert result.observation.prompt != ""

    def test_state_after_reset(self):
        env = ClinicalTriageEnv()
        env.reset(TaskID.EASY)
        s = env.state()
        assert s.step == 0
        assert s.done is False
        assert s.total_reward == 0.0

    def test_step_returns_valid_result(self):
        env = ClinicalTriageEnv()
        env.reset(TaskID.EASY)
        result = env.step(ClinicalAction(content="IMMEDIATE"))
        assert 0.0 <= result.reward <= 1.0
        assert isinstance(result.done, bool)
        assert result.observation is not None

    def test_step_after_done_raises(self):
        env = ClinicalTriageEnv()
        env.reset(TaskID.EASY)
        env.step(ClinicalAction(content="IMMEDIATE"))  # terminates on first valid
        with pytest.raises(RuntimeError):
            env.step(ClinicalAction(content="URGENT"))

    def test_reset_clears_state(self):
        env = ClinicalTriageEnv()
        env.reset(TaskID.EASY)
        env.step(ClinicalAction(content="IMMEDIATE"))
        env.reset(TaskID.MEDIUM)
        s = env.state()
        assert s.step == 0
        assert s.done is False
        assert s.task_id == TaskID.MEDIUM

    def test_all_tasks_reset(self):
        env = ClinicalTriageEnv()
        for task in [TaskID.EASY, TaskID.MEDIUM, TaskID.HARD]:
            result = env.reset(task)
            assert result.observation.task_id == task
            assert result.observation.prompt != ""

    def test_medium_observation_has_catalog(self):
        env = ClinicalTriageEnv()
        result = env.reset(TaskID.MEDIUM)
        assert result.observation.available_tests is not None
        assert len(result.observation.available_tests) > 0

    def test_hard_observation_has_diagnosis(self):
        env = ClinicalTriageEnv()
        result = env.reset(TaskID.HARD)
        assert result.observation.diagnosis is not None
        assert len(result.observation.diagnosis) > 5
