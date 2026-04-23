import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os

# 1. CARGA DEL MODELO
# Ajustamos la ruta para que sea robusta (asumiendo que está en la carpeta 'models')
ruta_modelo = os.path.join(os.path.dirname(__file__), 'models', 'model5_tuned_lightgbm_target_encoder.pkl')
model = joblib.load(ruta_modelo)

def predict_consumo(provincia, ano, normativa, tipo_edificio, superficie, compacidad, 
                    ventana_norte, ventana_sur, ventana_este, ventana_oeste, 
                    pct_calefactado, pct_refrigerado, tipo_generador, 
                    invierno, verano):
    
    # --- PROCESAMIENTO DE VARIABLES ---
    
    # 1. Creación de la Zona Climática Combinada (ej: A4)
    zona_climatica = f"{invierno}{verano}"
    
    # 2. Transformaciones Logarítmicas (Usamos np.log como en tu clean_data)
    # Ponemos un pequeño margen de 0.1 para evitar log(0) si los sliders bajaran a 0
    log_superficie = np.log(superficie if superficie > 0 else 0.1)
    log_compacidad = np.log(compacidad if compacidad > 0 else 0.1)
    
    # 3. Mapeo del Tipo de Generador (Agrupación según tu lógica)
    tipo_limpio = str(tipo_generador).lower().strip()
    if any(x in tipo_limpio for x in ['bomba', 'aire-aire', 'split', 'aerotermia']):
        tipo_agrupado = 'Bomba de Calor / Aerotermia'
    elif 'caldera' in tipo_limpio:
        tipo_agrupado = 'Calderas'
    elif 'rendimiento' in tipo_limpio and 'constante' in tipo_limpio:
        tipo_agrupado = 'Sistemas de Rendimiento Constante'
    elif 'joule' in tipo_limpio:
        tipo_agrupado = 'Efecto Joule'
    else:
        tipo_agrupado = 'Otros'

    # 4. Creación del DataFrame para el modelo
    # IMPORTANTE: El orden de estas columnas DEBE coincidir con el X_train del entrenamiento
    data = {
        'provincia': [provincia],
        'ano_construccion': [ano],
        'normativa': [normativa],
        'tipo_edificio': [tipo_edificio],
        'log_superficie': [log_superficie],
        'log_compacidad': [log_compacidad],
        'ventana_norte': [ventana_norte],
        'ventana_sur': [ventana_sur],
        'ventana_este': [ventana_este],
        'ventana_oeste': [ventana_oeste],
        'pct_calefactado': [pct_calefactado],
        'pct_refrigerado': [pct_refrigerado],
        'tipo_agrupado': [tipo_agrupado],
        'zona_climatica': [zona_climatica],
        'zona_clima_invierno': [invierno],
        'zona_clima_verano': [verano]
    }
    
    df_input = pd.DataFrame(data)

        # Extraemos el orden exacto de las columnas con las que se entrenó el pipeline

    columnas_entrenamiento = model.feature_names_in_
    
    # Reordenamos el DataFrame para que coincida al 100%
    df_input = df_input[columnas_entrenamiento]



    # 5. Predicción
    prediccion = model.predict(df_input)[0]
    
    return f"Consumo Estimado: {round(prediccion, 2)} kWh/m² año"

# --- CONFIGURACIÓN DE LA INTERFAZ GRADIO ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏢 Certificador Energético Andalucía AI")
    gr.Markdown("Introduce las características de la vivienda para estimar su consumo de energía.")
    
    with gr.Row():
        with gr.Column():
            provincia = gr.Dropdown(['Almería', 'Cádiz', 'Córdoba', 'Granada', 'Huelva', 'Jaén', 'Málaga', 'Sevilla'], label="Provincia", value="Sevilla")
            ano = gr.Number(label="Año de Construcción", value=2000)
            normativa = gr.Dropdown(['NBE-CT-79', 'Anterior', 'C.T.E.', 'Otros'], label="Normativa Vigente", value="C.T.E.")
            tipo_edificio = gr.Dropdown(['ViviendaUnifamiliar', 'ViviendaIndividualEnBloque', 'EdificioDeViviendas'], label="Tipo de Edificio", value="ViviendaIndividualEnBloque")
            
        with gr.Column():
            superficie = gr.Slider(31, 1000, label="Superficie Habitable (m²)", value=90)
            compacidad = gr.Slider(0.5, 10, label="Compacidad (m³/m²)", value=1.5)
            tipo_generador = gr.Dropdown(['Caldera', 'Bomba de Calor', 'Efecto Joule', 'Rendimiento Constante'], label="Sistema Calefacción", value="Bomba de Calor")

    with gr.Row():
        gr.Markdown("### 🪟 Configuración de Fachadas (% de Ventanas)")
    with gr.Row():
        v_norte = gr.Slider(0, 100, label="Norte", value=10)
        v_sur = gr.Slider(0, 100, label="Sur", value=20)
        v_este = gr.Slider(0, 100, label="Este", value=10)
        v_oeste = gr.Slider(0, 100, label="Oeste", value=10)

    with gr.Row():
        pct_cal = gr.Slider(0, 100, label="% Superficie Calefactada", value=100)
        pct_ref = gr.Slider(0, 100, label="% Superficie Refrigerada", value=50)
        inv = gr.Dropdown(['A', 'B', 'C', 'Otros'], label="Clima Invierno", value="B")
        ver = gr.Dropdown(['3', '4'], label="Clima Verano", value="3")

    btn = gr.Button("Calcular Consumo", variant="primary")
    resultado = gr.Textbox(label="Resultado de Predicción")

    btn.click(
        fn=predict_consumo,
        inputs=[provincia, ano, normativa, tipo_edificio, superficie, compacidad, 
                v_norte, v_sur, v_este, v_oeste, pct_cal, pct_ref, tipo_generador, inv, ver],
        outputs=resultado
    )

if __name__ == "__main__":
    demo.launch()