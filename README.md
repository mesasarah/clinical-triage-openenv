# 🏥 Clinical Triage Navigator — OpenEnv Environment

> **A real-world emergency department triage environment for training and evaluating AI agents.**  
> Built for the OpenEnv Round 1 Hackathon.

---

## Overview & Motivation

Every day, emergency departments around the world make high-stakes decisions under time pressure:  
*Who needs care right now? What tests should we run? Is this patient safe to go home?*

The **Clinical Triage Navigator** models this process as a structured reinforcement learning environment. An AI agent must demonstrate clinical reasoning across three progressively harder tasks — from simple urgency classification to full, safe discharge planning.

This environment fills a genuine gap in the OpenEnv ecosystem: **healthcare AI safety evaluation**. It is immediately useful for:

- Evaluating frontier LLMs on safety-critical reasoning
- Training agents that must balance caution and efficiency
- Benchmarking clinical decision support systems

---

## Environment Architecture

```
clinical-triage-env/
├── server/
│   ├── main.py          # FastAPI app — /reset, /step, /state, /health
│   ├── env.py           # Core environment state machine
│   └── models.py        # Typed Pydantic models (Observation, Action, Reward)
├── graders/
│   └── graders.py       # Deterministic graders for all 3 tasks
├── data/
│   └── fixtures.py      # Patient case fixtures + ground truth
├── tests/
│   └── test_env.py      # Full pytest suite
├── inference.py          # Baseline inference script
├── openenv.yaml          # OpenEnv metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Tasks

### Task 1 — Symptom Triage Classification (`task_easy`) ★☆☆

**Difficulty:** Easy  
**Max steps:** 5  

Given a patient's vitals, chief complaint, and history, the agent must classify urgency:

| Label | Meaning |
|---|---|
| `IMMEDIATE` | Life-threatening — act within seconds |
| `URGENT` | Needs assessment within 30 minutes |
| `SEMI_URGENT` | Needs assessment within 2 hours |
| `NON_URGENT` | Routine care |

**Action format:** One word — `IMMEDIATE`, `URGENT`, `SEMI_URGENT`, or `NON_URGENT`

**Reward:**
- Exact match → **1.0**
- Off by 1 level → **0.5**
- Off by 2 levels → **0.2**
- Off by 3+ levels → **0.0**
- Lethal error (e.g. `NON_URGENT` for an `IMMEDIATE` patient) → **0.0** + penalty flag

---

### Task 2 — Diagnostic Test Ordering (`task_medium`) ★★☆

**Difficulty:** Medium  
**Max steps:** 10  

Given a triaged patient and a catalog of 20 diagnostic tests, the agent orders the correct workup.

**Action format:** Comma-separated test codes — e.g. `CBC,CMP,ECG,TROPI`

**Reward components (total = 1.0):**
| Component | Weight | Description |
|---|---|---|
| Required coverage | 50% | Safety-critical tests must be present |
| Expected coverage | 30% | Proportion of recommended tests ordered |
| No over-ordering | 10% | Penalty for unnecessary tests |
| No forbidden tests | 10% | Hard penalty for contraindicated tests |

---

### Task 3 — Discharge Plan Generation (`task_hard`) ★★★

**Difficulty:** Hard  
**Max steps:** 15  

Given a patient ready for discharge, the agent produces a complete, clinically safe discharge plan across 5 mandatory sections: diagnosis, medications, follow-up, red-flag warnings, and activity/diet instructions.

**Action format:** Free-form structured text with all 5 sections

**Reward components (total = 1.0):**
| Component | Weight | Description |
|---|---|---|
| Diagnosis accuracy | 15% | Correct diagnosis documented |
| Medication coverage | 20% | Appropriate medications listed |
| Follow-up plan | 15% | Correct specialty, timeframe |
| Red-flag warnings | 25% | Must list ≥ N specific warning signs |
| Lifestyle instructions | 15% | Activity/diet guidance present |
| No forbidden content | 10% | No contraindicated drugs or instructions |

---

## Observation Space

```python
class ClinicalObservation(BaseModel):
    task_id:         str                        # which task is active
    step:            int                        # current step number
    patient:         PatientRecord              # full patient record
    available_tests: Optional[List[DiagnosticTest]]  # task_medium only
    ordered_tests:   Optional[List[str]]        # task_medium: already ordered
    test_results:    Optional[Dict[str, str]]   # test code → result
    diagnosis:       Optional[str]              # task_hard: confirmed diagnosis
    prompt:          str                        # natural-language instruction for agent
    episode_info:    Dict[str, Any]
```

---

## Action Space

```python
class ClinicalAction(BaseModel):
    content: str
    # task_easy:   "IMMEDIATE" | "URGENT" | "SEMI_URGENT" | "NON_URGENT"
    # task_medium: "CBC,CMP,ECG,TROPI"  (comma-separated test codes)
    # task_hard:   Full structured discharge plan text
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/reset` | Start new episode — body: `{"task_id": "task_easy"}` |
| `POST` | `/step` | Submit action — body: `{"content": "IMMEDIATE"}` |
| `GET` | `/state` | Current episode state (non-destructive) |

---

## Setup & Usage

### Local development

```bash
# Clone and install
git clone <your-repo-url>
cd clinical-triage-env
pip install -r requirements.txt

# Start server
python -m uvicorn server.main:app --host 0.0.0.0 --port 7860

# Run tests
pytest tests/ -v

# Run baseline
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
export HF_TOKEN="your-hf-token"
python inference.py
```

### Docker

```bash
docker build -t clinical-triage-env .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4.1-mini" \
  -e HF_TOKEN="your-hf-token" \
  clinical-triage-env
```

### Quick API test

```bash
# Health check
curl http://localhost:7860/health

# Start easy episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'

# Submit action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"content": "IMMEDIATE"}'

# Check state
curl http://localhost:7860/state
```

---

## Baseline Scores

Measured with `gpt-4.1-mini` at temperature 0, single-pass:

| Task | Score | Notes |
|---|---|---|
| `task_easy` — Triage Classification | **1.00** | Correct on clear-cut presentations |
| `task_medium` — Diagnostic Ordering | **0.73** | Misses some rare but expected tests |
| `task_hard` — Discharge Planning | **0.61** | Partial red-flag coverage; strong on medications |
| **Overall average** | **0.78** | |

---

## Reward Design Philosophy

This environment deliberately avoids **sparse, end-of-episode rewards**. Every task provides a graded signal:

- **Task Easy:** Partial credit for near-misses prevents the agent from being penalised equally for a 1-level error and a lethal 3-level error
- **Task Medium:** Four independent reward components mean the agent gets signal even with an incomplete order set
- **Task Hard:** Section-level scoring means the agent learns *which part* of the discharge plan needs improvement

**Penalties are explicit and interpretable** — the feedback string in every `StepResult.info` tells the agent exactly what it got wrong and why.

---

## Disqualification Safeguards

All graders are:
- ✅ Fully deterministic (same input → same score, always)
- ✅ Bounded to `[0.0, 1.0]`
- ✅ Resistant to prompt injection (text matching only, no model calls)
- ✅ Tested with a full pytest suite

---

## Hugging Face Spaces Deployment

This environment is deployed as a containerised HF Space tagged `openenv`.

**Space URL:** `(https://huggingface.co/spaces/sarahzephyr/clinical-triage-env)`

The Space uses the provided `Dockerfile` and exposes port `7860` as required.

---

## License

MIT License — see `LICENSE` for details.
