# -------------------------------------------------------------------
# Selected Feature Names embebidas (orden EXACTO del modelo)
# Se genera en los ficheros 'shap_xgboost_{days}_days_k{n_feat}_feat.txt
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
# Definición de features (en el orden de entrada al modelo)
# *** Usa exactamente los nombres que has pedido ***
# -------------------------------------------------------------------
features_24 = [
    "SIRS",
    "BUN Admission",
    "Pleural Effusion",
    "Creatinine Admission",
    "Inflamatory Index",
    "Monocytes Admission",
    "Alkaline phosphatase",   # f ALC
    "Albumin",
    "Hematocrit Admission",
    "Lipase",
    "Charlson Comorbidity Index (excluding age)",
    "White Blood Cell Admission",
    "Glucose",
    "C-reactive Protein Admission",
]

features_48 = [
    "SIRS",
    "BUN Admission",
    "CRP 48h",
    "Eosinophils 48h",
    "Lymphocytes 48h",
    "Creatinine Admission",
    "Pleural Effusion",
    "BUN 48h",
    "Albumin",
    "Inflamatory Index",
    "ALT",  #GPT
    "Monocytes Admission",
    "Creatinine 48h",
    "Intravenous fluids 48h",
    "White Blood Cell 48h",
    "Lipase",
    "Charlson Comorbidity Index (excluding age)",
    "Alkaline phosphatase",   # f ALC
    "Monocytes 48h",
    "Smoking status",  #packs/year
    "Waist circumference/Triglycerides"  #waist circunferece
]



# -------------------------------------------------------------------
# Mapeos opcionales UI -> nombres del modelo (.pkl/.txt)
# (ajusta aquí si los nombres en el modelo difieren de los de la UI)
# -------------------------------------------------------------------
ui_to_model_24 = {
    "C-reactive Protein Admission": "CRP Adm",
    "Glucose": "Glu",
    "White Blood Cell Admission": "WBC Adm",
    "Charlson Comorbidity Index (excluding age)": "CCI",
    "Hematocrit Admission": "Hct Adm",
    "Albumin": "Albumin",
    "Monocytes Admission": "Mono Adm",
    "Inflamatory Index": "Inflammatory Index",
    "Creatinine Admission": "Creat Adm",
    "Pleural Effusion": "Pleural Effusion",
    "BUN Admission": "BUN Adm",
    "SIRS": "SIRS",
    "Lipase": "Lipase",
    "Alkaline phosphatase": "F alc",
}

ui_to_model_48 = {
    "SIRS": "SIRS",
    "Monocytes 48h": "Mono 48h",
    "White Blood Cell 48h": "WBC 48h",
    "Intravenous fluids 48h": "Fluid 48h",
    "Creatinine 48h": "Creat 48h",
    "Creatinine Admission": "Creat Adm",
    "Lymphocytes 48h": "LYM 48h",
    "Eosinophils 48h": "Eosi 48h",
    "Monocytes Admission": "Mono Adm",
    "ALT": "GPT",
    "Inflamatory Index": "Inflammatory Index",
    "BUN 48h": "BUN 48h",
    "Pleural Effusion": "Pleural Effusion",
    "Albumin": "Albumin",
    "Lipase": "Lipase",
    "Alkaline phosphatase": "F alc",
    "Charlson Comorbidity Index (excluding age)": "CCI",
    "CRP 48h": "CRP 48h",
    "BUN Admission": "BUN Adm",
    "Smoking status": "Packs/Year",
    "Waist circumference/Triglycerides": "Waist circ",

}


# ---- 1) Define un mapa global de escalas (en nombres del MODELO) ----
# Usa factores multiplicativos (ej.: mg/dL -> mmol/L, etc.). Ajusta los que correspondan.
SCALE_MAP = {
    # --- Ejemplos frecuentes (ajusta a tus unidades reales) ---
    # Química:
    "Glucose": 1.0,            # mg/dL (si viniera en mmol/L: *18)
    "Creat Adm": 1.0,          # si UI está en mg/dL y el modelo espera µmol/L: *88.4
    "Creat 48h": 1.0,          # idem creatinina
    "BUN Adm": 1.0,            # mg/dL → mmol/L: *0.357 (si aplica)
    "BUN 48h": 1.0,
    "CRP Adm": 1.0,            # mg/L; si UI está en mg/dL: *10
    "CRP 48h": 1.0,
    "Lipase": 1.0,             # U/L; si UI la da en U/dL: *100
    "Albumin": 1.0,            # g/dL; si UI en g/L: *0.1
    # Hematología:
    "WBC Adm": 1000.0,            # modelo en normal, ui en x103
    "WBC 48h": 1000.0,
    "Platelets": 1000.0,          # (x103/μL)
    "Hct Adm": 0.01,           # % → fracción (ej. 42% -> 0.42)
    # Diferenciales:
    "Mono Adm": 1.0,
    "Mono 48h": 1.0,
    "LYM 48h": 1.0,
    "Eosi 48h": 1.0,
    # Otros:
    "Fluid 48h": 1000.0,        #modelo en mm, ui en L
    "Pleural Effusion": 1.0,   # binaria 0/1
    "SIRS": 1.0,               # binaria 0/1
    "F alc": 1.0,              # define si es binaria o cantidad
    "CCI": 1.0,
    "GPT": 1.0,
    "Inflammatory Index": 1000.0,  # x 1000
    "Waist circ": 1.0,
}

# Metadatos (pon aquí tus valores reales)
META = {
    # ===================== 24 h =====================
    "SIRS": {
        "binary": True,
        "unit": "",
        "hint": "0 = No, 1 = Yes"
    },
    "BUN Admission": {
        "unit": "mg/dL", "min": 1, "max": 200, "step": 1,
        "hint": "[1–200]"
    },
    "Pleural Effusion": {
        "binary": True,
        "unit": "",
        "hint": "0 = No, 1 = Yes"
    },
    "Creatinine Admission": {
        "unit": "mg/dL", "min": 0.2, "max": 15, "step": 0.01,
        "hint": "[0.2–15]"
    },
    "Inflamatory Index": {
        "unit": "×10^3", "min": 0, "max": 500, "step": 1,
        "hint": "[0–500]"
    },
    "Monocytes Admission": {
        "unit": "/μL", "min": 0, "max": 3000, "step": 10,
        "hint": "[0–3,000]"
    },
    "Alkaline phosphatase": {   # Nota: clínicamente se usa U/L
        "unit": "U/L", "min": 20, "max": 1000, "step": 1,
        "hint": "[20–1,000]"
    },
    "Albumin": {
        "unit": "g/dL", "min": 1, "max": 6, "step": 0.1,
        "hint": "[1.0–6.0]"
    },
    "Hematocrit Admission": {
        "unit": "%", "min": 10, "max": 70, "step": 0.1,
        "hint": "[10–70]"
    },
    "Lipase": {
        "unit": "U/L", "min": 0, "max": 5000, "step": 1,
        "hint": "[0–5,000]"
    },
    "Charlson Comorbidity Index (excluding age)": {
        "unit": "", "min": 0, "max": 30, "step": 1,
        "hint": "[0–30]"
    },
    "White Blood Cell Admission": {
        "unit": "×10^3/μL", "min": 0.5, "max": 40, "step": 0.1,
        "hint": "[0.5–40]"
    },
    "Glucose": {
        "unit": "mg/dL", "min": 40, "max": 600, "step": 1,
        "hint": "[40–600]"
    },
    "C-reactive Protein Admission": {  # asumo mg/L
        "unit": "mg/L", "min": 0, "max": 500, "step": 1,
        "hint": "[0–500]"
    },

    # ===================== 48 h =====================
    "CRP 48h": {  # asumo mg/L
        "unit": "mg/L", "min": 0, "max": 500, "step": 1,
        "hint": "[0–500]"
    },
    "Eosinophils 48h": {
        "unit": "/μL", "min": 0, "max": 2000, "step": 10,
        "hint": "[0–2,000]"
    },
    "Lymphocytes 48h": {
        "unit": "×10^3/μL", "min": 0, "max": 10, "step": 0.1,
        "hint": "[0–10]"
    },
    "BUN 48h": {
        "unit": "mg/dL", "min": 1, "max": 200, "step": 1,
        "hint": "[1–200]"
    },
    "Creatinine 48h": {
        "unit": "mg/dL", "min": 0.2, "max": 15, "step": 0.01,
        "hint": "[0.2–15]"
    },
    "White Blood Cell 48h": {
        "unit": "×10^3/μL", "min": 0.5, "max": 40, "step": 0.1,
        "hint": "[0.5–40]"
    },
    "Intravenous fluids 48h": {
        "unit": "L", "min": 0, "max": 10, "step": 0.1,
        "hint": "[0–10]"
    },
    "Monocytes 48h": {  # nota: /uL vs /μL
        "unit": "/μL", "min": 0, "max": 3000, "step": 10,
        "hint": "[0–3,000]"
    },
    "Platelets": {  # clínicamente ×10^3/μL
        "unit": "×10^3/μL", "min": 10, "max": 1500, "step": 1,
        "hint": "[10–1,500]"
    },
    "Waist circumference": {
        "unit": "cm", "min": 40, "max": 200, "step": 0.5,
        "hint": "[40–200]"
    },

    # Repetidas/compartidas (48 h)
    "SIRS": {  # ya definida arriba; si tu código usa dict.update, esta puede omitirse
        "binary": True,
        "unit": "",
        "hint": "0 = No, 1 = Yes"
    },
    "BUN Admission": {  # ya definida arriba
        "unit": "mg/dL", "min": 1, "max": 200, "step": 1,
        "hint": "[1–200]"
    },
    "Pleural Effusion": {  # ya definida arriba
        "binary": True,
        "unit": "",
        "hint": "0 = No, 1 = Yes"
    },
    "Creatinine Admission": {  # ya definida arriba
        "unit": "mg/dL", "min": 0.2, "max": 15, "step": 0.01,
        "hint": "[0.2–15]"
    },
    "Albumin": {  # ya definida arriba
        "unit": "g/dL", "min": 1, "max": 6, "step": 0.1,
        "hint": "[1.0–6.0]"
    },
    "Inflamatory index": {  # ojo: nombre distinto al de 24h
        "unit": "×10^3", "min": 0, "max": 500, "step": 1,
        "hint": "[0–500]"
    },
    "Monocytes Admission": {  # ya definida arriba
        "unit": "/μL", "min": 0, "max": 3000, "step": 10,
        "hint": "[0–3,000]"
    },
    "ALT": {
        "unit": "U/L", "min": 0, "max": 1000, "step": 1,
        "hint": "ALT/GPT [0–1,000]"
    },
    "Lipase": {  # ya definida arriba
        "unit": "U/L", "min": 0, "max": 5000, "step": 1,
        "hint": "[0–5,000]"
    },
    "Alkaline phosphatase": {  # nombre 48h con U/L
        "unit": "U/L", "min": 20, "max": 1000, "step": 1,
        "hint": "[20–1,000]"
    },
    "Charlson Comorbidity Index (excluding age)": {  # ya arriba
        "unit": "", "min": 0, "max": 30, "step": 1,
        "hint": "[0–30]"
    },
    "Smoking status": {
        "unit": "Packs/year", "min": 0, "max": 200, "step": 1,
        "hint": "[0–200]"
    },
}
