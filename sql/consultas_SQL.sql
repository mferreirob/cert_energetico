-- Vamos a ejecutar unas cuantas consultas para confirmar nuestras hipótesis antes de meternos a hacer los gráficos con Tableau

-- 1. Mucha diferencia entre norvativas...probablemente correlacionada con año de construccion

	-- 1.1 Para ello, empezamos revisando el consumo por normativa

SELECT 
    normativa,
    COUNT(*) as num_certificados,
    ROUND(AVG(consumo_global)::numeric, 2) as avg_consumo,
    ROUND(STDDEV(consumo_global)::numeric, 2) as desviacion_consumo,
    -- Revertimos el logaritmo y aseguramos que el resultado sea razonable
    ROUND(AVG(EXP(log_superficie))::numeric, 2) as avg_superficie_m2,
    ROUND(AVG(EXP(log_compacidad))::numeric, 2) as avg_compacidad_real,
    ROUND(AVG(ventana_sur)::numeric, 2) as avg_ventana_sur
FROM df_analisis_bi
WHERE 
    normativa IS NOT NULL 
    AND log_superficie IS NOT NULL 
    AND log_compacidad IS NOT NULL
    AND consumo_global IS NOT null
    and normativa is not null
    -- Filtro de seguridad por si se coló algún outlier extremo en el BI
    AND consumo_global > 0 
    AND consumo_global < 1000
GROUP BY normativa
ORDER BY avg_consumo DESC;


	-- 1.2 Ahora, miraremos si las normativas se corresponden con grupos de años aproximadamente

SELECT 
    normativa,
    COUNT(*) as total,
    ROUND(AVG(ano_construccion)::numeric, 0) as año_medio,
    MIN(ano_construccion) as año_min,
    MAX(ano_construccion) as año_max,
    ROUND(STDDEV(ano_construccion)::numeric, 2) as dispersion_años
FROM df_analisis_bi
WHERE normativa IS NOT NULL
GROUP BY normativa
ORDER BY año_medio ASC;

	-- 1.3 Ahora, agrupamos por decadas y ordenamos por consumo

SELECT 
    (FLOOR(ano_construccion / 10) * 10) as decada,
    COUNT(*) as volumen_viviendas,
    ROUND(AVG(consumo_global)) as avg_consumo_decada
FROM df_analisis_bi
WHERE 
    ano_construccion >= 1900 
    AND ano_construccion <= 2026
    AND consumo_global > 0 
    AND consumo_global < 1000
GROUP BY decada
ORDER BY decada;

-- 2. La compacidad, la superficie y el clima son los otros factores con más importancia

	-- 2.1 Compacidad

SELECT 
    ROUND(EXP(log_compacidad)) as compacidad_real,
    ROUND(AVG(consumo_global)) as avg_consumo,
    COUNT(*) as num_casas
FROM df_analisis_bi
WHERE log_compacidad IS NOT NULL AND consumo_global < 1000
GROUP BY compacidad_real
HAVING COUNT(*) > 50 -- Filtramos grupos pequeños para evitar ruido
ORDER BY compacidad_real ASC;

	--2.1 Superficie

SELECT 
    CASE 
        WHEN EXP(log_superficie) < 50 THEN '1. Muy Pequeña (<50m2)'
        WHEN EXP(log_superficie) BETWEEN 50 AND 100 THEN '2. Pequeña (50-100m2)'
        WHEN EXP(log_superficie) BETWEEN 100 AND 200 THEN '3. Mediana (100-200m2)'
        ELSE '4. Grande (>200m2)'
    END as rango_superficie,
    COUNT(*) as total_viviendas,
    ROUND(AVG(consumo_global)) as avg_consumo
FROM df_analisis_bi
WHERE log_superficie IS NOT NULL
GROUP BY rango_superficie
ORDER BY rango_superficie;

	-- 2.2 Clima

SELECT 
    zona_clima_invierno,
    COUNT(*) as certificados,
    ROUND(AVG(consumo_global)) as consumo_medio,
    ROUND(AVG(EXP(log_superficie))) as superficie_media
FROM df_analisis_bi
WHERE zona_clima_invierno IS NOT NULL
GROUP BY zona_clima_invierno
ORDER BY zona_clima_invierno ASC;
