
import pandas as pd
import joblib
import os
from src.processing import clean_data
from src.evaluation import saca_metricas
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder,OrdinalEncoder
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

mlflow.set_experiment("Certificado_Energetico_Regresion")

nombre_modelo = "model5_tuned_lightgbm_target_encoder"

with mlflow.start_run(run_name=nombre_modelo):
    # Autolog registra parámetros de LightGBM y métricas de sklearn
    mlflow.sklearn.autolog()

    # 0. ASEGURAR CARPETA DE MODELOS
    if not os.path.exists('models'):
        os.makedirs('models')

    # 1. CARGA DE DATOS
    print("Cargando datos...")
    df = pd.read_csv('data/clean/df_regression.csv')

    # 2. LIMPIEZA Y PREPROCESAMIENTO
    print("Limpiando y preprocesando datos...")
    data_cleaned = clean_data(df)

    # Definir X e y
    X = data_cleaned.drop(columns=['consumo_global', 'id_edificio'])
    y = data_cleaned['consumo_global']

    # DEFINICIÓN DE COLUMNAS PARA EL PIPELINE
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # CONSTRUCCIÓN DEL PREPROCESADOR (ColumnTransformer)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', TargetEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. PIPELINE COMPLETO CON MODELO

    # Como vimos en el AutoML previo, los modelos que mejor funcionan en este dataset son los de boosting. Vamos a probar varias configuraciones de de lightGBM y XGBoost.

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LGBMRegressor(
            n_estimators=1500, 
            learning_rate=0.05, 
            max_depth=11, 
            random_state=42, 
            n_jobs=-1))
        ])

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. ENTRENAMIENTO

    print(f"Entrenando Pipeline con {len(X_train)} filas...")
    model_pipeline.fit(X_train, y_train)


    # 5. EVALUACIÓN

    preds = model_pipeline.predict(X_test)

    r2 = r2_score(y_test, preds)

    # BASELINE (media del train)
    baseline_value = np.mean(y_train)
    baseline_preds = np.full_like(y_test, baseline_value)
    mae_baseline = mean_absolute_error(y_test, baseline_preds)
    mae = mean_absolute_error(y_test, preds)

    print("\n" + "="*30)
    print(f"R2 SCORE: {r2:.4f}")
    print(f"MAE modelo: {mae:.4f} kWh/m2")
    print(f"MAE baseline (media): {mae_baseline:.4f} kWh/m2")

    improvement = (mae_baseline - mae) / mae_baseline * 100
    print(f"Mejora vs baseline: {improvement:.2f}%")
    print("="*30)

    mlflow.log_metric("mae_baseline", mae_baseline)
    mlflow.log_metric("improvement_vs_baseline_pct", improvement)

    # 6. IMPORTANCIAS

    model = model_pipeline.named_steps['regressor']
    importances = model.feature_importances_     #///En caso de que no estemos entrenando un ensemble
    #importances = model_pipeline.named_steps['regressor'].named_estimators_['xgb'].feature_importances_  #///En caso de que estemos entrenando un ensemble

    # Sacar los nombres tras el preprocesamiento
    feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()

    feature_imp_df = pd.DataFrame({'Variable': feature_names, 'Importancia': importances})

    feature_imp_df = feature_imp_df.sort_values(by='Importancia', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 8)) # Un poco más grande para que quepan bien los nombres
    sns.barplot(
        x='Importancia', 
        y='Variable', 
        data=feature_imp_df, 
        palette='viridis' # La paleta 'viridis' queda genial y es muy legible
    )

    plt.title(f'Ranking de Importancia de las Variables ({nombre_modelo})', fontsize=16)
    plt.xlabel('Importancia (Gain / Split)', fontsize=12)
    plt.ylabel('Variables del Modelo', fontsize=12)
    plt.tight_layout() 

    plot_path = "feature_importance_ordenado.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)

    os.remove(plot_path)

    # 7. GUARDAR PIPELINE (Incluye preprocesamiento + modelo)
    joblib.dump(model_pipeline, f'models/{nombre_modelo}.pkl')
    print("Pipeline guardado con éxito.")

    mlflow.sklearn.log_model(model_pipeline, nombre_modelo)

    print("¡Todo registrado en MLflow!")