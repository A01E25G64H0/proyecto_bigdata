# Sistema de Detección de Fraude Financiero

Plataforma analítica y predictiva *end-to-end* diseñada para procesar más de **6.36 millones de transacciones financieras**, entrenar modelos de aprendizaje automático sobre clases severamente desbalanceadas y exponer un servicio de inferencia en tiempo real consumido por una aplicación web interactiva.

**Autor:** A. Enrique Guzmán Hernández

---

## Enlaces en Producción

| Componente | Tecnología | URL |
| :--- | :--- | :--- |
| **Aplicación Web (Frontend)** | Streamlit Cloud | https://proyectobigdata-hxn4gmezjkkxy5uwm8nlyo.streamlit.app |
| **Documentación Interactiva** | Swagger UI / OpenAPI | https://api-fraude-paysim.onrender.com/docs |

> **Nota sobre tiempos de respuesta:** El backend se encuentra desplegado en una instancia gratuita de Render que entra en suspensión tras 15 minutos de inactividad. La primera petición puede demorar entre **45 y 60 segundos** mientras inicializa el contenedor y carga el modelo en memoria; las consultas subsecuentes responden en milisegundos.

---

## Arquitectura General y Estructura del Repositorio

El sistema implementa una arquitectura desacoplada en cuatro etapas: procesamiento distribuido en Databricks, registro de experimentos en MLflow, despliegue de microservicio contenedorizado en Render y visualización ligera en Streamlit Cloud (sin cargar archivos `.pkl` localmente en el frontend).

```text
[Databricks Lakehouse: Bronze → Silver → Gold (6.36M registros)]
                                │
                 [MLflow: RandomForest_Balanced]
                                │
          [API REST FastAPI + Pydantic en Render (.pkl)]
                                │
                    (Peticiones HTTP / JSON)
                                │
        [Frontend Streamlit Cloud: Evaluador + Tablero Gold]
```

### Estructura de Carpetas

```text
proyecto_bigdata/
├── databricks_notebooks/   # Notebooks ETL (Bronze, Silver, Gold) y entrenamiento MLflow
├── api_service/            # Servicio FastAPI, esquemas Pydantic, Dockerfile y modelo (.pkl)
├── frontend_app/           # Aplicación web desacoplada en Streamlit
│   ├── app.py              # Evaluador de riesgo en vivo y tablero de monitoreo
│   ├── gold_kpis.csv       # Indicadores agregados exportados desde la Capa Gold
│   └── requirements.txt    # Dependencias ligeras (streamlit, requests, pandas, plotly)
└── README.md               # Documentación técnica del proyecto
```

---

## 1. Limpieza y Transformación de Datos (Arquitectura Medallion)

El procesamiento de datos se ejecuta en Databricks mediante un flujo orquestado en *Jobs & Pipelines* (`01_bronze_ingestion` → `02_silver_cleaning` → `03_gold_feature_engineering`) sobre tablas gestionadas en formato **Delta Lake**:

* **Capa Bronze (`default.bronze_onlinefraud`):** Ingesta cruda del conjunto de datos PaySim (`onlinefraud.csv`) conservando la estructura original de los **6,362,604 registros** e incorporando columnas de auditoría (`ingestion_timestamp` y `source_file`) para garantizar trazabilidad.
* **Capa Silver (`default.silver_onlinefraud`):**
  * *Deduplicación y nulos:* Eliminación de registros duplicados exactos (`step`, `type`, `amount`, `nameOrig`, `nameDest`) y filtrado de valores nulos en campos críticos (`type`, `amount`, `isFraud`).
  * *Estandarización de tipos y limpieza:* Conversión estricta a tipos numéricos de precisión (`double` para montos y saldos; `int` para pasos temporales y banderas), creación de la variable binaria `is_merchant_dest` para identificar cuentas destino comerciales (prefijo `'M'`) y descarte de identificadores o columnas sin varianza (`nameOrig`, `nameDest`, `isFlaggedFraud`).
  * *Tratamiento de outliers y escala:* Filtrado de operaciones con `amount > 0`, aplicación de Winsorización al percentil 99 (`P99 = 1,623,278.90`) para acotar valores extremos sin perder la señal de riesgo y transformación logarítmica (`log_amount = log1p(amount_capped)`).
* **Capa Gold (`default.gold_onlinefraud` y `default.gold_kpis`):**
  * *Ingeniería de características:* Extracción de la hora del día (`hour_of_day = step % 24`) y cálculo de discrepancias contables entre saldos iniciales, montos transferidos y saldos finales tanto en origen (`errorBalanceOrig`) como en destino (`errorBalanceDest`).
  * *Agregación de negocio:* Generación de la vista analítica `default.gold_kpis` (exportada a `frontend_app/gold_kpis.csv`) con métricas consolidadas por hora y tipo de operación.

### Indicadores Globales Procesados (Capa Gold)

| Métrica | Valor Agregado |
| :--- | :--- |
| **Transacciones Analizadas** | 6,362,604 |
| **Fraudes Detectados** | 8,197 (~0.13% del total) |
| **Monto Total en Riesgo** | $12,056,415,427.84 |

---

## 2. Definición del Problema, Entrenamiento y Métricas

* **Tipo de tarea:** Clasificación binaria supervisada (`isFraud` $\in \{0, 1\}$) bajo un escenario de desbalanceo severo de clases, donde solo el **0.13%** de las operaciones corresponden a fraude real.
* **Estrategia de muestreo y preprocesamiento:** En el notebook `04_ml_training_mlflow`, se conserva el **100% de las transacciones fraudulentas** junto con una muestra aleatoria del **8% de operaciones legítimas** (división estratificada 80/20). El preprocesamiento se encapsula en un `Pipeline` de Scikit-Learn (`ColumnTransformer`) que aplica `OneHotEncoder` a la variable categórica `type` y `StandardScaler` a las variables numéricas, serializando todo el flujo dentro de `fraud_model.pkl`.
* **Justificación de métricas:** Dado el desbalanceo, la exactitud global (*Accuracy*) resulta engañosa (un modelo trivial que prediga siempre `0` obtendría 99.87% sin detectar un solo fraude). Por ello se seleccionaron:
  * **F1-Score (métrica principal):** Media armónica entre *Precision* y *Recall*, indispensable para equilibrar la captura efectiva de fraudes reales sin generar exceso de falsas alarmas que bloqueen clientes legítimos.
  * **PR-AUC (Área bajo la curva Precision-Recall):** Evalúa el desempeño del clasificador específicamente sobre la clase minoritaria a través de distintos umbrales de decisión.

### Comparación de Experimentos en MLflow (`/Shared/Fraude_Financiero_PaySim`)

| Modelo | F1-Score | PR-AUC | Precision | Recall | Tiempo de Entrenamiento |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `LogisticRegression_Balanced` | 0.370 | 0.807 | 0.230 | 0.952 | 13.9 s |
| **`RandomForest_Balanced` (Seleccionado)** | **0.998** | **0.998** | **0.999** | **0.997** | **35.7 s** |

Se seleccionó **`RandomForest_Balanced`** para despliegue productivo debido a su capacidad para capturar interacciones no lineales entre el tipo de operación y los descuadres de saldo, alcanzando un **F1-Score de 0.9982** y **PR-AUC de 0.9978**.

---

## 3. Publicación del Modelo (API REST)

El *pipeline* entrenado se expone mediante un microservicio en **FastAPI** empaquetado con **Docker** (`python:3.10-slim` + Uvicorn en el puerto `8000`) y alojado en Render:

* **Validación de entradas:** Utiliza esquemas **Pydantic** (`schemas.py`) para verificar tipos de datos, campos obligatorios y rangos válidos antes de ejecutar la inferencia, devolviendo errores descriptivos (`HTTP 422`) ante peticiones malformadas.
* **Transformación en tiempo de inferencia:** El endpoint calcula dinámicamente `hour_of_day`, aplica el tope de Winsorización (`P99_CAP = 1623278.90`) con `log1p` para obtener `log_amount` y deriva los errores de balance (`errorBalanceOrig` y `errorBalanceDest`) antes de consultar el modelo.
* **Monitoreo y documentación:** Incluye `GET /health` para verificación de estado y `GET /docs` (Swagger UI) para pruebas interactivas.

### Esquema del Endpoint `POST /predict`

**Entrada (JSON):**

| Campo | Tipo | Restricciones de Validación |
| :--- | :--- | :--- |
| `step` | `int` | Entre `1` y `744` (hora de simulación) |
| `type` | `str` | `"TRANSFER"`, `"CASH_OUT"`, `"PAYMENT"`, `"CASH_IN"` o `"DEBIT"` |
| `amount` | `float` | Mayor a `0` y menor o igual a `10,000,000` |
| `oldbalanceOrg` | `float` | Mayor o igual a `0` |
| `newbalanceOrig` | `float` | Mayor o igual a `0` |
| `oldbalanceDest` | `float` | Mayor o igual a `0` |
| `newbalanceDest` | `float` | Mayor o igual a `0` |
| `is_merchant_dest` | `int` | `0` (cliente) o `1` (comercio) |

**Salida (JSON):** Devuelve `is_fraud` (`0` o `1`), `fraud_probability` (probabilidad entre `0.0` y `1.0`) y `decision` (`"APROBAR"` si la probabilidad es $< 0.5$ o `"BLOQUEAR"` si es $\ge 0.5$).

---

## 4. Integración con la Aplicación Web y Casos de Prueba

La aplicación en **Streamlit Cloud** (`frontend_app/app.py`) opera exclusivamente como cliente HTTP mediante `requests`, sin dependencias de entrenamiento ni archivos `.pkl` locales. Implementa manejo explícito de excepciones (`Timeout` configurado a **90 segundos** para tolerar el arranque en frío de Render, `ConnectionError`, `HTTPError` y `ValueError` por respuestas no JSON).

### Ejemplo 1: Transacción Legítima (`APROBAR`)

* **Método y Endpoint:** `POST https://api-fraude-paysim.onrender.com/predict`
* **Interpretación:** Coherencia contable tanto en el descuento del emisor (`45,000 → 41,500`) como en el abono al receptor (`8,000 → 11,500`).

**Body (Request):**

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
```

**Resultado (Response):**

```json
{
  "is_fraud": 0,
  "fraud_probability": 0.0173,
  "decision": "APROBAR"
}
```

### Ejemplo 2: Toma de Cuenta / Fraude (`BLOQUEAR`)

* **Método y Endpoint:** `POST https://api-fraude-paysim.onrender.com/predict`
* **Interpretación:** El atacante vacía el 100 % del saldo disponible (`25,000 → 0`) hacia una cuenta destino que no refleja el abono (`0 → 0`).

**Body (Request):**

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
```

**Resultado (Response):**

```json
{
  "is_fraud": 1,
  "fraud_probability": 1.0,
  "decision": "BLOQUEAR"
}
```
