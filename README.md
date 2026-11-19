# Sistema de Detección de Fraude Financiero en Tiempo Real (PaySim)

Plataforma analítica y predictiva *end-to-end* diseñada para procesar más de **6.36 millones de transacciones financieras**, entrenar modelos de aprendizaje automático sobre clases desbalanceadas y exponer un servicio de inferencia en tiempo real consumido por una aplicación web interactiva.

**Autor(es):** [Escribe aquí tu nombre completo / matrícula]

---

## Enlaces en Producción

* **Aplicación Web (Frontend en Streamlit Cloud):** https://proyectobigdata-hxn4gmezjkkxy5uwm8nlyo.streamlit.app
* **API REST de Inferencia (Backend en Render):** https://api-fraude-paysim.onrender.com
* **Endpoint de Predicción (`POST`):** https://api-fraude-paysim.onrender.com/predict
* **Documentación Interactiva (Swagger UI / OpenAPI):** https://api-fraude-paysim.onrender.com/docs

> **Nota sobre tiempos de respuesta:** El backend se encuentra desplegado en una instancia gratuita de Render que entra en suspensión tras 15 minutos de inactividad. La primera petición puede demorar entre 45 y 60 segundos mientras inicializa el contenedor y carga el modelo en memoria; las consultas subsecuentes responden en milisegundos.

---

## Arquitectura General y Estructura del Repositorio

El sistema implementa una arquitectura desacoplada en cuatro etapas: procesamiento distribuido en **Databricks**, registro de experimentos en **MLflow**, despliegue de microservicio en **Render** y visualización ligera en **Streamlit Cloud** (sin cargar archivos `.pkl` localmente en el frontend).

[Databricks Lakehouse: Bronze → Silver → Gold (6.36M registros)]
                               │
                 [MLflow: RandomForest_Balanced]
                               │
           [API REST FastAPI + Pydantic en Render (.pkl)]
                               │
                    (Peticiones HTTP / JSON)
                               │
         [Frontend Streamlit Cloud: Evaluador + Tablero Gold]

Estructura de carpetas

proyecto_bigdata/
├── databricks_notebooks/   # Notebooks ETL (Bronze, Silver, Gold) y entrenamiento MLflow
├── api_service/            # Servicio FastAPI, esquemas Pydantic y artefacto del modelo (.pkl)
├── frontend_app/           # Aplicación web desacoplada en Streamlit
│   ├── app.py              # Evaluador de riesgo en vivo y tablero de monitoreo
│   ├── gold_kpis.csv       # Indicadores agregados exportados desde la Capa Gold
│   └── requirements.txt    # Dependencias ligeras (streamlit, requests, pandas, plotly)
└── README.md               # Documentación técnica del proyecto


# 1. Limpieza y Transformación de Datos (Arquitectura Medallion)

El procesamiento de datos se ejecuta en Databricks mediante un flujo orquestado en Jobs & Pipelines (01_bronze_ingestion  02_silver_cleaning  03_gold_feature_engineering) sobre tablas gestionadas en formato Delta Lake:

Capa Bronze (bronze_onlinefraud): Ingesta cruda del conjunto de datos PaySim conservando la estructura original de los 6,362,604 registros para garantizar trazabilidad y auditoría.

Capa Silver (silver_onlinefraud):
Nulos y duplicados: Validación de integridad referencial, eliminación de registros duplicados exactos e imputación/filtrado de valores nulos críticos en montos y saldos.

Tipos de dato: Estandarización estricta a tipos numéricos de precisión (double para montos y saldos, int para pasos temporales y banderas) y cadenas limpias (string) para categorías operativas.

Tratamiento de outliers: En detección de fraude financiero, las transferencias de montos extremos constituyen señales críticas de toma de cuenta (account takeover), por lo que se conservaron como eventos válidos en lugar de recortarse estadísticamente.

Variables derivadas: Extracción de la hora del día (hour_of_day a partir de step) y creación de la bandera binaria is_merchant_dest para identificar cuentas destino comerciales (prefijo 'M').

Capa Gold (gold_onlinefraud y gold_kpis):
Ingeniería de características y codificación: Cálculo de discrepancias contables entre saldos iniciales, montos transferidos y saldos finales tanto en origen como en destino, además de la codificación de la variable categórica type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER).

Agregación de negocio: Generación de la vista analítica gold_kpis (exportada a frontend_app/gold_kpis.csv) con métricas consolidadas por hora y tipo de operación.

Indicadores Globales Procesados (Capa Gold)

Métrica                              Valor Agregado
Transacciones Analizadas                6,362,604
Fraudes Detectados                 8,197 (~0.13% del total)
Monto Total en Riesgo              $12,056,415,427.84

# 2. Definición del Problema y Métricas de Evaluación

Tipo de tarea: Clasificación binaria supervisada (is_fraud {0,1}) bajo un escenario de desbalanceo severo de clases (solo el 0.13% de las operaciones corresponden a fraude real).

Justificación de métricas: Dado el desbalanceo, la exactitud global (Accuracy) resulta engañosa (un modelo trivial que prediga siempre 0 obtendría 99.87% de exactitud sin detectar un solo fraude). Por ello se seleccionaron:

F1-Score (métrica principal): Media armónica entre Precision y Recall, indispensable para equilibrar la captura efectiva de fraudes reales sin generar exceso de falsas alarmas que bloqueen clientes legítimos.

PR-AUC (Área bajo la curva Precision-Recall): Evalúa el desempeño del clasificador específicamente sobre la clase minoritaria (fraude) a través de distintos umbrales de decisión.

Comparación de Experimentos en MLflow (/Shared/Fraude_Financiero_PaySim)

Desde el notebook 04_ml_training_mlflow se entrenaron y registraron dos algoritmos con balanceo de pesos de clase:

Modelo                       F1-Score         PR-AUC       Precision        Recall          Tiempo de Entrenamiento 

Logistic Regression_Balanced 
                                0.370          0.807         0.230           0.952                       13.9 s


Random Forest_Balanced
(Seleccionado)                  0.998           0.998         0.999          0.997                        35.7 s
               
Se seleccionó RandomForest_Balanced para despliegue productivo debido a su capacidad para capturar interacciones no lineales entre el tipo de operación y los descuadres de saldo, alcanzando un F1-Score de 0.9982 y PR-AUC de 0.9978.

# 3. Publicación del Modelo (API REST)
El modelo entrenado se expone mediante un microservicio en FastAPI alojado en Render (https://api-fraude-paysim.onrender.com)
Validación de entradas: Utiliza esquemas Pydantic para verificar tipos de datos, campos obligatorios y rangos válidos antes de ejecutar la inferencia, devolviendo errores descriptivos (HTTP 422) ante peticiones malformadas.

Documentación OpenAPI: Accesible de forma interactiva en /docs (Swagger UI).

Esquema del Endpoint POST /predict
Entrada (JSON): step (int), type (str), amount (float), oldbalanceOrg (float), newbalanceOrig (float), oldbalanceDest (float), newbalanceDest (float), is_merchant_dest (int).

Salida (JSON): is_fraud (0 o 1), fraud_probability (probabilidad entre 0.0 y 1.0) y decision ("APROBAR" o "BLOQUEAR").

# 4. Integración con la Aplicación Web y Casos de Prueba

La aplicación en Streamlit Cloud (`frontend_app/app.py`) opera exclusivamente como cliente HTTP (`requests`), sin dependencias de entrenamiento ni archivos `.pkl` locales. Implementa manejo explícito de excepciones (`Timeout` configurado a 90s para tolerar el arranque en frío de Render, `ConnectionError`, `HTTPError` y `ValueError` por respuestas no JSON).

# Ejemplo 1: Transacción Legítima (`APROBAR`)

* **Método y Endpoint:** `POST [https://api-fraude-paysim.onrender.com/predict](https://api-fraude-paysim.onrender.com/predict)`
* **Interpretación:** Coherencia contable tanto en el descuento del emisor (`45,000` → `41,500`) como en el abono al receptor (`8,000` → `11,500`).

**Body:**
```json
{
  "step": 10,
  "type": "TRANSFER",
  "amount": 3500,
  "oldbalanceOrg": 45000,
  "newbalanceOrig": 41500,
  "oldbalanceDest": 8000,
  "newbalanceDest": 11500,
  "is_merchant_dest": 0
}

# Resultado:

JSON
{
  "is_fraud": 0,
  "fraud_probability": 0.0173,
  "decision": "APROBAR"
}

# Ejemplo 2: Toma de Cuenta / Fraude (BLOQUEAR)
* **Método y Endpoint:** POST [https://api-fraude-paysim.onrender.com/predict](https://api-fraude-paysim.onrender.com/predict)

* **Interpretación:** El atacante vacía el 100 % del saldo disponible (25,000 → 0) hacia una cuenta destino que no refleja el abono (0 → 0)[cite: 13].

**Body:**
```json
{
  "step": 10,
  "type": "TRANSFER",
  "amount": 25000,
  "oldbalanceOrg": 25000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 0,
  "newbalanceDest": 0,
  "is_merchant_dest": 0
}

# Resultado:

JSON
{
  "is_fraud": 1,
  "fraud_probability": 1.0,
  "decision": "BLOQUEAR"
}
