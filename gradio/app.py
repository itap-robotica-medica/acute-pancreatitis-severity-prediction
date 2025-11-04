import gradio as gr
import utils
from definitions import features_24, features_48, META


def make_number(label_name: str):
    meta = META.get(label_name, {})
    unit = meta.get("unit")
    lbl = f"{label_name}" + (f" ({unit})" if unit else "")
    return gr.Number(
        label=lbl,
        value=0.0,
        minimum=meta.get("min"),
        maximum=meta.get("max"),
        step=meta.get("step"),
        info=meta.get("hint"),   # se muestra en gris bajo el campo
    )


# -------------------------------------------------------------------
# Construcción de la UI con dos zonas 50/50
# -------------------------------------------------------------------
with gr.Blocks(title="Predicción — 24h & 48h",
               theme="default",
               css="""
               .titulo-centro {text-align: center; margin-bottom: 20px;}
                .subtitulo-centro {text-align: center; font-size: 20px; font-weight: 600; color: #2B6CB0; margin-bottom: 12px;}
                .footer {text-align: center; font-size: 14px; color: gray; margin-top: 20px; border-top: 1px solid #ddd; padding-top: 10px;}
                :root { --radius-xl: 14px; }
                .wrap-col { border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; }
                """
    ) as demo:
    gr.Markdown(
        """
        # 🧠 Early Prediction of Acute Pancreatitis (AP) Severity 
        ### SHAP-based XGBoost Model using 24h and 48h clinical variables
        """,
        elem_classes = ["titulo-centro"]
    )

    with gr.Row(equal_height=True):
        # -------------------- Zona Izquierda (24h) --------------------
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["wrap-col"]):
                gr.Markdown("<h3 style='text-align: center;'>24-hour model</h3>")

                inputs_24 = [make_number(name) for name in features_24]

                btn_24 = gr.Button("Calcular predicción (24h)", variant="primary")
                out_24 = gr.Number(label="AP probability (%):", precision=2, interactive=False)

        # -------------------- Zona Derecha (48h) --------------------
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["wrap-col"]):
                gr.Markdown("<h3 style='text-align: center;'>48-hour model</h3>")

                inputs_48 = [make_number(name) for name in features_48]

                btn_48 = gr.Button("Calcular probabilidad (48h)", variant="primary")
                out_48 = gr.Number(label="AP probability (%):", precision=2, interactive=False)

    gr.Markdown(
        """
        © 2025 — Based on Cisnal, A. Ruiz-Rebollo, M.L. et al.</i>,  
        <b>Improved Early Prediction of Acute Pancreatitis Severity Using SHAP-Based XGBoost Model: Beyond Traditional Scoring Systems</b>.  
        DOI: <a href="https://doi.org/10.xxxx/yyyy" target="_blank">10.xxxx/yyyy</a>
        """,
        elem_classes=["footer"]
    )

    # Enlaces
    btn_24.click(fn=utils.predict_24, inputs=inputs_24, outputs=out_24)
    btn_48.click(fn=utils.predict_48, inputs=inputs_48, outputs=out_48)

if __name__ == "__main__":
    # server_name="0.0.0.0"
    demo.launch()
