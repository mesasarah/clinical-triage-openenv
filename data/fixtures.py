"""
Deterministic patient case fixtures used across all three tasks.
Ground-truth labels are embedded here for reproducible grading.
"""
from __future__ import annotations
from typing import List, Dict, Any
from server.models import PatientRecord, Vitals, DiagnosticTest


# ──────────────────────────────────────────────
# Diagnostic Test Catalog (used in task_medium)
# ──────────────────────────────────────────────

DIAGNOSTIC_CATALOG: List[DiagnosticTest] = [
    DiagnosticTest(code="CBC",    name="Complete Blood Count",          category="haematology",    description="Full blood panel including WBC, RBC, haemoglobin, haematocrit, platelets"),
    DiagnosticTest(code="CMP",    name="Comprehensive Metabolic Panel", category="chemistry",      description="Electrolytes, glucose, BUN, creatinine, liver enzymes"),
    DiagnosticTest(code="ECG",    name="12-lead Electrocardiogram",     category="cardiology",     description="Electrical activity of the heart"),
    DiagnosticTest(code="TROPI",  name="Troponin I (high-sensitivity)", category="cardiology",     description="Cardiac injury marker; elevated in MI"),
    DiagnosticTest(code="CXR",    name="Chest X-Ray",                   category="radiology",      description="Plain film of chest: lungs, heart, mediastinum"),
    DiagnosticTest(code="CTHEAD", name="CT Head (non-contrast)",        category="radiology",      description="Brain parenchyma, haemorrhage, mass lesion"),
    DiagnosticTest(code="CTPA",   name="CT Pulmonary Angiogram",        category="radiology",      description="Pulmonary embolism workup"),
    DiagnosticTest(code="US_ABD", name="Abdominal Ultrasound",          category="radiology",      description="Liver, gallbladder, pancreas, kidneys, aorta"),
    DiagnosticTest(code="LFTX",   name="Liver Function Tests",          category="chemistry",      description="ALT, AST, ALP, GGT, bilirubin, albumin"),
    DiagnosticTest(code="LIPASE", name="Serum Lipase",                  category="chemistry",      description="Elevated in pancreatitis"),
    DiagnosticTest(code="LACTATE",name="Serum Lactate",                 category="chemistry",      description="Tissue hypoperfusion marker; elevated in sepsis/shock"),
    DiagnosticTest(code="BCULX2", name="Blood Cultures x2",            category="microbiology",   description="Aerobic and anaerobic; for bacteraemia/sepsis"),
    DiagnosticTest(code="UA",     name="Urinalysis + Microscopy",       category="microbiology",   description="Infection, haematuria, proteinuria"),
    DiagnosticTest(code="UCULX",  name="Urine Culture",                 category="microbiology",   description="Pathogen identification for UTI/pyelonephritis"),
    DiagnosticTest(code="DDIMER", name="D-Dimer",                       category="haematology",    description="Fibrin degradation; sensitive but non-specific for VTE"),
    DiagnosticTest(code="BNP",    name="B-type Natriuretic Peptide",    category="cardiology",     description="Heart failure marker"),
    DiagnosticTest(code="ABG",    name="Arterial Blood Gas",            category="respiratory",    description="pH, pO2, pCO2, HCO3; respiratory/metabolic status"),
    DiagnosticTest(code="BHCG",   name="Serum Beta-hCG",               category="endocrinology",  description="Pregnancy test; ectopic workup in females"),
    DiagnosticTest(code="TSH",    name="Thyroid Stimulating Hormone",   category="endocrinology",  description="Thyroid function"),
    DiagnosticTest(code="COAG",   name="Coagulation Screen (PT/APTT)",  category="haematology",    description="Clotting function"),
]

CATALOG_BY_CODE: Dict[str, DiagnosticTest] = {t.code: t for t in DIAGNOSTIC_CATALOG}


# ──────────────────────────────────────────────
# Task Easy — 5 triage cases
# ──────────────────────────────────────────────

EASY_CASES: List[Dict[str, Any]] = [
    {
        "patient": PatientRecord(
            patient_id="E001",
            age=58,
            sex="Male",
            chief_complaint="Central crushing chest pain radiating to the left arm for 40 minutes",
            history="Known hypertension, type 2 diabetes, smoker 20 pack-years. Diaphoretic and pale on arrival.",
            vitals=Vitals(heart_rate=112, blood_pressure="88/60", respiratory_rate=22, temperature_c=36.8, spo2_percent=94, gcs=15),
            allergies=["penicillin"],
            current_meds=["metformin 500mg BD", "amlodipine 5mg OD"],
        ),
        "ground_truth": "IMMEDIATE",
        "rationale": "STEMI presentation with haemodynamic compromise — act in seconds",
    },
    {
        "patient": PatientRecord(
            patient_id="E002",
            age=34,
            sex="Female",
            chief_complaint="Severe right-sided headache with photophobia and neck stiffness since this morning",
            history="No prior headaches. Vomited twice en route. No fever recorded at home.",
            vitals=Vitals(heart_rate=98, blood_pressure="135/85", respiratory_rate=18, temperature_c=38.9, spo2_percent=98, gcs=14),
            allergies=[],
            current_meds=[],
        ),
        "ground_truth": "IMMEDIATE",
        "rationale": "Meningism + fever + altered GCS — bacterial meningitis until proven otherwise",
    },
    {
        "patient": PatientRecord(
            patient_id="E003",
            age=45,
            sex="Male",
            chief_complaint="Sudden-onset severe right loin pain radiating to groin, unable to get comfortable",
            history="First episode. No haematuria noted. Nausea present.",
            vitals=Vitals(heart_rate=95, blood_pressure="148/90", respiratory_rate=16, temperature_c=37.1, spo2_percent=99, gcs=15),
            allergies=["NSAIDs"],
            current_meds=[],
        ),
        "ground_truth": "URGENT",
        "rationale": "Classic renal colic — painful but not immediately life-threatening; needs analgesia and imaging within 30 min",
    },
    {
        "patient": PatientRecord(
            patient_id="E004",
            age=22,
            sex="Female",
            chief_complaint="Sore throat and mild fever for 2 days, difficulty swallowing",
            history="No stridor. Mouth opens normally. Tonsils erythematous, no peritonsillar bulge.",
            vitals=Vitals(heart_rate=88, blood_pressure="118/76", respiratory_rate=14, temperature_c=38.2, spo2_percent=99, gcs=15),
            allergies=[],
            current_meds=[],
        ),
        "ground_truth": "SEMI_URGENT",
        "rationale": "Probable tonsillitis — no airway compromise; needs assessment within 2 hours",
    },
    {
        "patient": PatientRecord(
            patient_id="E005",
            age=67,
            sex="Male",
            chief_complaint="Mild ankle swelling for three weeks, no pain at rest",
            history="Hypertension, on ramipril. No chest pain or dyspnoea. Bilateral pitting oedema to mid-shin.",
            vitals=Vitals(heart_rate=72, blood_pressure="138/82", respiratory_rate=14, temperature_c=36.5, spo2_percent=98, gcs=15),
            allergies=[],
            current_meds=["ramipril 5mg OD"],
        ),
        "ground_truth": "NON_URGENT",
        "rationale": "Chronic oedema, haemodynamically stable — routine assessment appropriate",
    },
]


# ──────────────────────────────────────────────
# Task Medium — 3 ordering scenarios
# ──────────────────────────────────────────────

MEDIUM_CASES: List[Dict[str, Any]] = [
    {
        "patient": PatientRecord(
            patient_id="M001",
            age=62,
            sex="Male",
            chief_complaint="Acute chest pain and shortness of breath",
            history="Hypertension, hyperlipidaemia. Pain started at rest 90 minutes ago, 8/10, radiates to jaw.",
            vitals=Vitals(heart_rate=108, blood_pressure="96/64", respiratory_rate=24, temperature_c=37.0, spo2_percent=92, gcs=15),
            allergies=[],
            current_meds=["atorvastatin 40mg", "lisinopril 10mg"],
        ),
        "triage": "IMMEDIATE",
        # Must-have: ECG, TROPI, CXR, CMP, CBC, COAG
        # Nice-to-have: ABG, BNP
        # Penalise if missing ECG or TROPI (dangerous omission)
        "required_tests":  ["ECG", "TROPI"],
        "expected_tests":  ["ECG", "TROPI", "CXR", "CMP", "CBC", "COAG"],
        "allowed_extras":  ["ABG", "BNP"],
        "forbidden_tests": ["BHCG", "TSH", "UCULX"],
        "rationale":       "Suspected NSTEMI/PE — ECG and troponin are safety-critical",
    },
    {
        "patient": PatientRecord(
            patient_id="M002",
            age=38,
            sex="Female",
            chief_complaint="Severe right upper quadrant pain, nausea, vomiting",
            history="Obese. Pain post-fatty meal, radiating to right shoulder. Murphy's sign positive.",
            vitals=Vitals(heart_rate=102, blood_pressure="124/78", respiratory_rate=18, temperature_c=38.4, spo2_percent=98, gcs=15),
            allergies=["morphine"],
            current_meds=[],
        ),
        "triage": "URGENT",
        "required_tests":  ["US_ABD", "LFTX"],
        "expected_tests":  ["US_ABD", "LFTX", "CBC", "CMP", "LIPASE"],
        "allowed_extras":  ["COAG"],
        "forbidden_tests": ["CTHEAD", "CTPA", "ECG", "TROPI"],
        "rationale":       "Cholecystitis — ultrasound and LFTs are the cornerstone workup",
    },
    {
        "patient": PatientRecord(
            patient_id="M003",
            age=71,
            sex="Male",
            chief_complaint="Confusion, fever, and decreased urine output for 24 hours",
            history="Type 2 diabetes, CKD stage 3. Daughter reports he has been unwell for 2 days with dysuria.",
            vitals=Vitals(heart_rate=118, blood_pressure="92/56", respiratory_rate=26, temperature_c=39.4, spo2_percent=95, gcs=13),
            allergies=[],
            current_meds=["metformin 1g BD", "furosemide 40mg OD"],
        ),
        "triage": "IMMEDIATE",
        "required_tests":  ["BCULX2", "LACTATE", "UA"],
        "expected_tests":  ["BCULX2", "LACTATE", "UA", "UCULX", "CBC", "CMP", "ABG"],
        "allowed_extras":  ["COAG", "LFTX"],
        "forbidden_tests": ["CTHEAD", "TSH", "BHCG"],
        "rationale":       "Urosepsis with shock — Sepsis-6 bundle; blood cultures and lactate are safety-critical",
    },
]


# ──────────────────────────────────────────────
# Task Hard — 2 full discharge scenarios
# ──────────────────────────────────────────────

HARD_CASES: List[Dict[str, Any]] = [
    {
        "patient": PatientRecord(
            patient_id="H001",
            age=55,
            sex="Female",
            chief_complaint="Epigastric pain, nausea, vomiting for 12 hours",
            history="Alcohol history (40 units/week). Serum lipase 1,200 U/L (3× ULN). CT abdomen: oedematous pancreas, no necrosis. Managed with IV fluids and analgesia. Now tolerating oral intake.",
            vitals=Vitals(heart_rate=84, blood_pressure="118/74", respiratory_rate=14, temperature_c=37.1, spo2_percent=99, gcs=15),
            allergies=[],
            current_meds=[],
        ),
        "diagnosis": "Acute mild pancreatitis secondary to alcohol",
        # Grader keyword sets — at least N of these must appear in the discharge plan
        "required_keywords": {
            "diagnosis":    ["pancreatitis"],
            "medications":  ["paracetamol", "ibuprofen", "ondansetron", "omeprazole"],
            "follow_up":    ["gastroenterology", "gp", "general practitioner", "one week", "2 weeks", "two weeks"],
            "red_flags":    ["fever", "worsening pain", "unable to eat", "jaundice", "vomiting"],
            "lifestyle":    ["alcohol", "abstain", "abstinence"],
        },
        "forbidden_keywords": ["opioid", "morphine", "codeine", "metformin"],
        "min_red_flags": 3,
    },
    {
        "patient": PatientRecord(
            patient_id="H002",
            age=29,
            sex="Female",
            chief_complaint="Right lower quadrant pain, fever",
            history="WBC 15.2 (elevated), CRP 98. CT abdomen: inflamed appendix, no perforation. Underwent laparoscopic appendicectomy. Day 1 post-op, tolerating fluids.",
            vitals=Vitals(heart_rate=78, blood_pressure="112/70", respiratory_rate=14, temperature_c=37.3, spo2_percent=99, gcs=15),
            allergies=["penicillin"],
            current_meds=[],
        ),
        "diagnosis": "Acute appendicitis — post laparoscopic appendicectomy day 1",
        "required_keywords": {
            "diagnosis":    ["appendicitis", "appendicectomy"],
            "medications":  ["paracetamol", "ibuprofen", "metronidazole"],
            "follow_up":    ["surgical", "surgeon", "one week", "1 week", "gp"],
            "red_flags":    ["fever", "wound", "redness", "swelling", "discharge", "pain worsening", "vomiting"],
            "activity":     ["avoid heavy lifting", "light activity", "driving"],
        },
        "forbidden_keywords": ["penicillin", "amoxicillin", "augmentin", "co-amoxiclav"],
        "min_red_flags": 4,
    },
]
