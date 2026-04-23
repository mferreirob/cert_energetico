import pandas as pd
import numpy as np
import os
import pprint
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder, OrdinalEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from src.processing import clean_data


# 1. CARGA Y PREPARACIÓN
print("Cargando datos...")
df = pd.read_csv('data/clean/df_regression.csv')
data_cleaned = clean_data(df)

# X e y 
X = data_cleaned.drop(columns=['consumo_global', 'id_edificio'])
y = data_cleaned['consumo_global']

# Muestreo de seguridad para evitar que pete la RAM 
if len(X) > 200000:
    X_train_full, _, y_train_full, _ = train_test_split(X, y, train_size=200000, random_state=42)
else:
    X_train_full, y_train_full = X, y

# 2. DEFINICIÓN DEL PREPROCESADOR 
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('target_enc', TargetEncoder())]), categorical_features)
])

# 3. EL PIPELINE COMO ESTIMADOR
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, n_jobs=-1, verbose=-1))
])

# 4. ESPACIO DE BÚSQUEDA (Con el prefijo regressor__)

param_dist = {
    'regressor__n_estimators': [500,1000,1500],
    'regressor__learning_rate': [0.05, 0.01, 0.1],
    'regressor__max_depth': [5,8,15]
}

# 5. CONFIGURACIÓN DEL BUSCADOR
print(f"🚀 Iniciando tuneo con {len(X_train_full)} registros...")

random_search = RandomizedSearchCV(
    estimator=full_pipeline, 
    param_distributions=param_dist,
    n_iter=10,
    scoring='r2',
    cv=3,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_full, y_train_full)

# 6. RESULTADOS
print("\n" + "="*50)
print("🏆 ¡MEJOR CONFIGURACIÓN ENCONTRADA!")
print("="*50)



# Extraemos los parámetros limpiando el prefijo 'regressor__'
mejores_params = {k.replace('regressor__', ''): v for k, v in random_search.best_params_.items()}

# Imprimimos de forma bonita
pprint.pprint(mejores_params)

print(f"\n⭐ Score R2 en Validación (CV): {random_search.best_score_:.4f}")
print("="*50)