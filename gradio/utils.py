import numpy as np
import joblib
from pathlib import Path
from typing import List, Tuple, re
from definitions import features_24, features_48, selected_24, selected_48, ui_to_model_24, ui_to_model_48
from definitions import SCALE_MAP, META

# -------------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------------
def load_or_dummy(path: Path, n_features: int):
    """
    Carga un modelo desde 'path'. Si no existe, crea uno de ejemplo con salida tipo probabilidad [0,1].
    """
    if path.exists():
        return joblib.load(path)
    # Modelo demo: prob = sigmoid(w·x + b) con pesos aleatorios estables
    rng = np.random.default_rng(42)
    w = rng.normal(0, 0.2, size=n_features)
    b = 0.0

    class DummyProbModel:
        def __init__(self, w, b):
            self.w = w
            self.b = b

        def predict_proba(self, X):
            X = np.asarray(X)
            z = X @ self.w + self.b
            prob = 1 / (1 + np.exp(-z))
            # emula sklearn predict_proba: columna clase 0 y 1
            return np.column_stack([1 - prob, prob])

        # compatibilidad mínima si alguien llama predict
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    return DummyProbModel(w, b)

def ensure_prob_0_1(model, X: np.ndarray) -> float:
    """
    Intenta obtener probabilidad de clase positiva en [0,1] con prioridad:
    1) predict_proba[:,1]
    2) decision_function -> sigmoid
    3) predict (0/1) -> ya está en [0,1]
    """
    # 1) predict_proba
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X.reshape(1, -1))[0, -1]
            return float(np.clip(proba, 0.0, 1.0))
        except Exception:
            pass

    # 2) decision_function -> sigmoid
    if hasattr(model, "decision_function"):
        try:
            score = model.decision_function(X.reshape(1, -1))
            # soporta salida escalar o array
            score = np.array(score).ravel()[0]
            prob = 1 / (1 + np.exp(-score))
            return float(np.clip(prob, 0.0, 1.0))
        except Exception:
            pass

    # 3) predict -> 0/1
    if hasattr(model, "predict"):
        try:
            pred = model.predict(X.reshape(1, -1))
            pred = float(np.array(pred).ravel()[0])
            return float(np.clip(pred, 0.0, 1.0))
        except Exception:
            pass

    # fallback seguro
    return 0.0


def build_vector_from_ui(values_list, ui_labels, selected_features, ui2model_map):
    """
    Mapea valores de la UI al orden exacto esperado por el modelo (selected_features).
    - ui2model_map resuelve diferencias de nombre (UI -> modelo).
    - Si falta alguna feature del modelo, se usa 0.0 y se avisa por consola.
    """
    ui_values = dict(zip(ui_labels, values_list))
    model_values = {}
    for ui_name, val in ui_values.items():
        model_name = ui2model_map.get(ui_name, ui_name)
        model_values[model_name] = float(val if val is not None else 0.0)

    # Aplica ESCALA por variable (independiente del modelo 24/48)
    for feat_name, factor in SCALE_MAP.items():
        if feat_name in model_values:
            model_values[feat_name] *= factor

    # Construye el vector en el orden exacto del modelo
    missing = []
    x = []
    for feat in selected_features:
        if feat in model_values:
            x.append(model_values[feat])
        else:
            x.append(0.0)
            missing.append(feat)
    if missing:
        print(f"[WARN] Features esperadas por el modelo ausentes en UI/mapeo: {missing}")
    return np.array(x, dtype=float)




# -------------------------------------------------------------------
# Carga de modelos (o dummy si no están)
# -------------------------------------------------------------------
model_24 = load_or_dummy(Path("shap_xgboost_1_days_k14_feat.pkl"), n_features=len(features_24))
model_48 = load_or_dummy(Path("shap_xgboost_12_days_k21_feat.pkl"), n_features=len(features_48))

# -------------------------------------------------------------------
# Funciones de predicción conectadas a la UI
# -------------------------------------------------------------------
def predict_24(*vals_24) -> float:
    x = build_vector_from_ui(
        values_list=vals_24,
        ui_labels=features_24,
        selected_features=selected_24,
        ui2model_map=ui_to_model_24,
    )
    prob = ensure_prob_0_1(model_24, x)
    return round(prob * 100.0, 2)

def predict_48(*vals_48) -> float:
    x = build_vector_from_ui(
        values_list=vals_48,
        ui_labels=features_48,
        selected_features=selected_48,
        ui2model_map=ui_to_model_48,
    )
    prob = ensure_prob_0_1(model_48, x)
    return round(prob * 100.0, 2)