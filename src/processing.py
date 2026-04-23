import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from NuestrasFunciones import *


def clean_data(df):

    df = df.copy()

    # Primeros filtros según lógica de negocio y modelo
    df = df.loc[(df['consumo_global'] > 0) & (df['consumo_global'] < 1000), :]
    df = df.loc[df['superficie'] > 30, :]
    df = df.loc[(df['compacidad'] > 0.5) & (df['compacidad'] < 100), :]
    df = df.reset_index(drop=True)

    # Transformaciones a logarítmicas
    df['log_superficie'] = np.log(df['superficie'])
    df['log_compacidad'] = np.log(df['compacidad'])

    # Reducción cardinalidad categóricas
    provincias = list(df.provincia.value_counts().head(8).index)
    df = df.loc[df['provincia'].isin(provincias), :]

    df.zona_climatica = df.zona_climatica.str.upper().str.strip()
    df['zona_clima_invierno'] = df.zona_climatica.apply(lambda x: x[0] if pd.notnull(x) else x)
    df['zona_clima_verano'] = df.zona_climatica.apply(lambda x: x[1] if pd.notnull(x) else x)
    df['zona_clima_invierno'] = df['zona_clima_invierno'].replace({'E': 'Otros'})
    df['zona_clima_invierno'] = df['zona_clima_invierno'].replace({'D': 'Otros'})
    df = df.loc[~df['zona_clima_verano'].isin(['1', '2']), :] 

    top_normativas = ['NBE-CT-79', 'Anterior', 'C.T.E.']
    df['normativa'] = df['normativa'].apply(lambda x: x if x in top_normativas else 'Otros')

    df['tipo_limpio'] = df['tipo_generador_cal'].astype(str).str.lower().str.strip()
    condiciones = [
        df['tipo_limpio'].str.contains(r'bomba|aire-aire|split|aerotermia', na=False),
        df['tipo_limpio'].str.contains(r'caldera', na=False),
        df['tipo_limpio'].str.contains(r'rendimiento.*constante', na=False),
        df['tipo_limpio'].str.contains(r'joule', na=False)
    ]
    nombres = [
        'Bomba de Calor / Aerotermia',
        'Calderas',
        'Sistemas de Rendimiento Constante',
        'Efecto Joule'
    ]
    df['tipo_agrupado'] = np.select(condiciones, nombres, default='Otros')

    # Eliminación columnas no deseadas
    df.drop(columns=['superficie'], inplace=True)
    df.drop(columns=['compacidad'], inplace=True)
    df.drop(columns=['municipio'], inplace=True)
    df = df.drop(columns=['tipo_limpio', 'tipo_generador_cal'])

    # Gestión outliers
    #varobjetivo = df['consumo_global']
    #imput = df.drop(columns=['consumo_global'])
    #impCont = imput.select_dtypes(include=np.number).copy()
    #imput_wins = impCont.apply(lambda x: gestiona_outliers(x,clas='winsor'))
    #imput_wins = pd.concat([imput_wins, imput.select_dtypes(exclude=np.number)], axis=1)

    # Gestión missings
    df['normativa'] = df['normativa'].fillna('desconocido')

    # Conversión categóricas
    #var_cat = df_imputed.select_dtypes(exclude=np.number).nunique().sort_values(ascending=False).index.tolist()
    #dummies = pd.get_dummies(df_imputed[var_cat])
    #df_dummies = pd.concat([df_imputed.drop(columns=var_cat), dummies], axis=1)
    #df_depurado =df_dummies.set_index('id_edificio')

    return df