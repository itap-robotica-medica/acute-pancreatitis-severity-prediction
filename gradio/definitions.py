# -------------------------------------------------------------------
# Selected Feature Names  (excat order from the model)
# It is genereted from the code files: 'shap_xgboost_{days}_days_k{n_feat}_feat.txt
# -------------------------------------------------------------------
selected_24 = [
    'CRP Adm', 'Glu', 'WBC Adm', 'CCI', 'Lipase', 'Hct Adm', 'Albumin',
    'F alc', 'Mono Adm', 'Inflammatory Index', 'Creat Adm', 'Pleural Effusion',
    'BUN Adm', 'SIRS'
]

selected_48 = [
    'Waist circ', 'Packs/Year', 'Mono 48h', 'F alc', 'CCI', 'Lipase',
    'WBC 48h', 'Fluid 48h', 'Creat 48h', 'Mono Adm', 'GPT', 'Inflammatory Index',
    'Albumin', 'BUN 48h', 'Pleural Effusion', 'Creat Adm', 'LYM 48h', 'Eosi 48h',
    'CRP 48h', 'BUN Adm', 'SIRS'
]


# -------------------------------------------------------------------
# UI feature lists
# -------------------------------------------------------------------
features_24 = [
    "SIRS",
    "BUN Admission",
    "Pleural Effusion",
    "Creatinine Admission",
    "Inflamatory Index",
    "Monocytes Admission",
    "Alkaline phosphatase (ALP)",   # f ALC
    "Albumin",
    "Hematocrit Admission",
    "Lipase",
    "Charlson Comorbidity Index (excluding age)",
    "White Blood Cell (WBC) Admission",
    "Glucose",
    "C-reactive Protein (CRP) Admission",
]

features_48 = [
    "SIRS",
    "Blood Urea Nitrogen (BUN) Admission",
    "C-reactive protein (CRP) 48h",
    "Eosinophils 48h",
    "Lymphocytes 48h",
    "Creatinine Admission",
    "Pleural Effusion",
    "Blood Urea Nitrogen (BUN) 48h",
    "Albumin",
    "Inflamatory Index",
    "Alanine Aminotransferase (ALT)",   #alt --> GPT
    "Monocytes Admission",
    "Creatinine 48h",
    "Intravenous fluids 48h",
    "White Blood (WBC) Cell 48h",
    "Lipase",
    "Charlson Comorbidity Index (excluding age)",
    "Alkaline phosphatase (ALP)",   # f ALC
    "Monocytes 48h",
    "Smoking status",  #packs/year
    "Waist circumference/Triglycerides"  #waist circunferece
]



# -------------------------------------------------------------------
# UI -> Model name maps (name differences)
# -------------------------------------------------------------------
ui_to_model_24 = {
    "C-reactive Protein (CRP) Admission": "CRP Adm",
    "Glucose": "Glu",
    "White Blood Cell (WBC) Admission": "WBC Adm",
    "Charlson Comorbidity Index (excluding age)": "CCI",
    "Hematocrit Admission": "Hct Adm",
    "Albumin": "Albumin",
    "Monocytes Admission": "Mono Adm",
    "Inflamatory Index": "Inflammatory Index",
    "Creatinine Admission": "Creat Adm",
    "Pleural Effusion": "Pleural Effusion",
    "Blood Urea Nitrogen (BUN) Admission": "BUN Adm",
    "SIRS": "SIRS",
    "Lipase": "Lipase",
    "Alkaline phosphatase (ALP)": "F alc",
}

ui_to_model_48 = {
    "SIRS": "SIRS",
    "Monocytes 48h": "Mono 48h",
    "White Blood Cell (WBC) 48h": "WBC 48h",
    "Intravenous fluids 48h": "Fluid 48h",
    "Creatinine 48h": "Creat 48h",
    "Creatinine Admission": "Creat Adm",
    "Lymphocytes 48h": "LYM 48h",
    "Eosinophils 48h": "Eosi 48h",
    "Monocytes Admission": "Mono Adm",
    "Alanine Aminotransferase (ALT)": "GPT",
    "Inflamatory Index": "Inflammatory Index",  # NETs adm / Platets = (PMN Adm/ LYM adm )/platets
    "BUN 48h": "BUN 48h",
    "Pleural Effusion": "Pleural Effusion",
    "Albumin": "Albumin",
    "Lipase": "Lipase",
    "Alkaline phosphatase (ALP)": "F alc",
    "Charlson Comorbidity Index (excluding age)": "CCI",
    "C-reactive Protein (CRP) 48h": "CRP 48h",
    "Blood Urea Nitrogen (BUN) Admission": "BUN Adm",
    "Smoking status": "Packs/Year",
    "Waist circumference/Triglycerides": "Waist circ",

}


# ---- 1) Define un mapa global de escalas (en nombres del MODELO) ----
# Usa factores multiplicativos (ej.: mg/dL -> mmol/L, etc.). Ajusta los que correspondan.
# Escalas por NOMBRE DE MODELO: UI -> escala del modelo
SCALE_MAP = {
    # --- Química / Bioquímica (misma unidad UI↔modelo, salvo indicación) ---
    "Glu": 1.0,               # mg/dL
    "Creat Adm": 1.0,         # mg/dL
    "Creat 48h": 1.0,         # mg/dL
    "BUN Adm": 1.0,           # mg/dL
    "BUN 48h": 1.0,           # mg/dL
    "CRP Adm": 1.0,           # mg/L  (PCR Adm)
    "CRP 48h": 1.0,           # mg/L  (PCR 48h)
    "Lipase": 1.0,            # U/L
    "Albumin": 1.0,           # g/dL
    "GPT": 1.0,               # U/L (UI 'ALT' -> modelo 'GPT')
    "F alc": 1.0,             # U/L (UI 'Alkaline phosphatase' -> modelo 'F alc')

    # --- Hemato / Diferenciales ---
    "WBC Adm": 1000.0,        # UI: ×10^3/μL  → modelo: /μL (Leukocytes Adm)
    "WBC 48h": 1000.0,        # UI: ×10^3/μL  → modelo: /μL (Leukocytes 48h)
    "Hct Adm": 0.01,          # UI: %         → modelo: fracción
    "Mono Adm": 1.0,          # UI: /μL       → modelo: /μL
    "Mono 48h": 1.0,          # UI: /μL       → modelo: /μL
    "LYM 48h": 1000.0,        # UI: ×10^3/μL  → modelo: /μL   (ACTUALIZADO)
    "Eosi 48h": 1.0,          # UI: /μL       → modelo: /μL
    "Platelets": 1000.0,      # (si usas UI en ×10^3/μL) → modelo: /μL (tabla en ~221000)

    # --- Otros ---
    "Fluid 48h": 1000.0,      # UI: Litros    → modelo: mL
    "Pleural Effusion": 1.0,  # binaria 0/1
    "SIRS": 1.0,              # binaria 0/1
    "CCI": 1.0,               # índice
    "Inflammatory Index": 1000.0,  # UI: ×10^3 → modelo: unidades absolutas (tabla ~1.7e6)
    "Waist circ": 1.0,        # cm (UI 'Waist circumference/Triglycerides' -> modelo 'Waist circ')
    "Packs/Year": 1.0,        # UI 'Smoking status' -> modelo 'Packs/Year'
}


# ========= META (común 24/48 h) =========
# Etiquetas = exactamente las usadas en la UI (features_24 / features_48)
META = {
    # --- Binarias / indicadores ---
    "SIRS": {"binary": True, "unit": "", "hint": "0 = No, 1 = Yes"},
    "Pleural Effusion": {"binary": True, "unit": "", "hint": "0 = No, 1 = Yes"},

    # --- Química / Bioquímica ---
    "Blood Urea Nitrogen (BUN) Admission": {"unit": "mg/dL", "min": 1, "max": 200, "value": 1, "step": 1, "hint": "[1–200]"},
    "Blood Urea Nitrogen (BUN) 48h": {"unit": "mg/dL", "min": 1, "max": 200, "value": 1, "step": 1, "hint": "[1–200]"},
    "Creatinine Admission": {"unit": "mg/dL", "min": 0.2, "max": 15, "value": 0.2, "step": 0.01, "hint": "[0.2–15]"},
    "Creatinine 48h": {"unit": "mg/dL", "min": 0.2, "max": 15, "value": 0.2, "step": 0.01, "hint": "[0.2–15]"},
    "C-reactive Protein (CRP) Admission": {"unit": "mg/L", "min": 0, "max": 500, "step": 1, "hint": "[0–500]"},
    "C-reactive Protein (CRP) 48h": {"unit": "mg/L", "min": 0, "max": 500, "step": 1, "hint": "[0–500]"},
    "Glucose": {"unit": "mg/dL", "min": 40, "max": 600, "value": 40, "step": 1, "hint": "[40–600]"},
    "Lipase": {"unit": "U/L", "min": 0, "max": 5000, "step": 1, "hint": "[0–5,000]"},
    "Albumin": {"unit": "g/dL", "min": 1, "max": 6, "value": 1, "step": 0.1, "hint": "[1.0–6.0]"},
    "Alanine Aminotransferase (ALT)": {"unit": "U/L", "min": 0, "max": 1000, "step": 1, "hint": "[0–1,000]"},
    "Alkaline phosphatase (ALP)": {"unit": "U/L", "min": 20, "max": 1000, "value": 20, "step": 1, "hint": "[20–1,000]"},
    "Intravenous fluids 48h": {"unit": "L", "min": 0, "max": 20, "step": 0.1, "hint": "[0–20]"},


    # --- Hematología / Diferenciales ---
    "White Blood Cell (WBC) Admission": {"unit": "×10^3/μL", "min": 0.5, "max": 40, "value": 0.5, "step": 0.1, "hint": "[0.5–40]"},
    "White Blood Cell (WBC) 48h": {"unit": "×10^3/μL", "min": 0.5, "max": 40, "step": 0.1, "hint": "[0.5–40]"},
    "Hematocrit Admission": {"unit": "%", "min": 10, "max": 70, "value": 10, "step": 0.1, "hint": "[10–70]"},
    "Lymphocytes 48h": {"unit": "×10^3/μL", "min": 0, "max": 10, "step": 0.1, "hint": "[0–10]"},
    "Eosinophils 48h": {"unit": "/μL", "min": 0, "max": 2000, "step": 10, "hint": "[0–2,000]"},
    "Monocytes Admission": {"unit": "/μL", "min": 0, "max": 3000, "step": 10, "hint": "[0–3,000]"},
    "Monocytes 48h": {"unit": "/μL", "min": 0, "max": 3000, "step": 10, "hint": "[0–3,000]"},

    # --- Índices / Antropometría / Hábitos (numéricos en tu UI) ---
    "Inflamatory Index": {"unit": "×10^3", "min": 0, "max": 5000, "step": 1, "hint": "[0–5000]"},
    "Smoking status": {"unit": "Packs/year", "min": 0, "max": 800, "step": 1, "hint": "[0–800]"},
    "Waist circumference/Triglycerides": {"unit": "cm", "min": 40, "max": 200, "value": 40, "step": 0.5, "hint": "[40–200]"},
}

