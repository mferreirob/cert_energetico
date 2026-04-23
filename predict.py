import joblib
import pandas as pd
import numpy as np
import os

# 1. CARGA DEL MODELO
# Usamos una ruta relativa robusta para que funcione en cualquier carpeta
ruta_actual = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(ruta_actual, 'models', 'model5_tuned_lightgbm_target_encoder.pkl')

try:
    pipeline = joblib.load(model_path)
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    exit()

def realizar_prediccion(datos_usuario):
    """
    Recibe un diccionario con los datos, aplica transformaciones
    y devuelve la predicción reordenando las columnas automáticamente.
    """
    # Crear DataFrame inicial
    df_input = pd.DataFrame([datos_usuario])

    # 2. TRANSFORMACIONES DE INGENIERÍA DE DATOS
    # Sincronizado con tu función clean_data y la lógica de la API
    
    # Logaritmos
    df_input['log_superficie'] = np.log(df_input['superficie'])
    df_input['log_compacidad'] = np.log(df_input['compacidad'])
    
    # Zona Climática Combinada
    df_input['zona_climatica'] = df_input['zona_clima_invierno'].astype(str) + df_input['zona_clima_verano'].astype(str)
    
    # Mapeo del Tipo de Generador (Tipo Agrupado)
    tipo_limpio = str(datos_usuario.get('tipo_generador_cal', '')).lower().strip()
    if any(x in tipo_limpio for x in ['bomba', 'aire-aire', 'split', 'aerotermia']):
        df_input['tipo_agrupado'] = 'Bomba de Calor / Aerotermia'
    elif 'caldera' in tipo_limpio:
        df_input['tipo_agrupado'] = 'Calderas'
    elif 'rendimiento' in tipo_limpio and 'constante' in tipo_limpio:
        df_input['tipo_agrupado'] = 'Sistemas de Rendimiento Constante'
    elif 'joule' in tipo_limpio:
        df_input['tipo_agrupado'] = 'Efecto Joule'
    else:
        df_input['tipo_agrupado'] = 'Otros'

    # 3. REORDENADO AUTOMÁTICO SEGÚN EL MODELO
    # Extraemos el orden exacto de las columnas con las que se entrenó el pipeline

    columnas_entrenamiento = pipeline.feature_names_in_
    
    # Reordenamos el DataFrame para que coincida al 100%
    df_final = df_input[columnas_entrenamiento]
    # 4. PREDICCIÓN FINAL
    pred = pipeline.predict(df_final)[0]
    return round(pred, 2)

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    # Datos de prueba (puedes cambiarlos para testear diferentes escenarios)
    casa_test = {
        'provincia': 'Granada',
        'ano_construccion': 1975,
        'normativa': 'Anterior',
        'tipo_edificio': 'ViviendaUnifamiliar',
        'superficie': 120.0,
        'compacidad': 1.2,
        'ventana_norte': 40.0,
        'ventana_sur': 10.0,
        'ventana_este': 5.0,
        'ventana_oeste': 15.0,
        'pct_calefactado': 80.0,
        'pct_refrigerado': 0.0,
        'tipo_generador_cal': 'Caldera gasoil antigua', # Esto se mapeará a 'Calderas'
        'zona_clima_invierno': 'D',
        'zona_clima_verano': '3'
    }

    resultado = realizar_prediccion(casa_test)
    
    print("\n" + "="*42)
    print("🏠  RESULTADO DE LA AUDITORÍA ENERGÉTICA AI")
    print("="*42)
    print(f" Provincia:      {casa_test['provincia']}")
    print(f" Año:            {casa_test['ano_construccion']}")
    print(f" Clima:          {casa_test['zona_clima_invierno']}{casa_test['zona_clima_verano']}")
    print("-" * 42)
    print(f" CONSUMO ESTIMADO: {resultado} kWh/m² año")
    print("="*42 + "\n")