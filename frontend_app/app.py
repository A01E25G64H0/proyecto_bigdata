import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Detección de Fraude PaySim", layout="wide")

# Reemplaza por la URL pública de tu API en Render o Hugging Face
DEFAULT_API_URL = "https://tu-servicio-fraude.onrender.com/predict"

st.title("Sistema de Detección de Fraude Financiero en Tiempo Real")
api_url = st.sidebar.text_input("Endpoint de la API REST:", value=DEFAULT_API_URL)

tab1, tab2 = st.tabs(["Evaluador de Riesgo en Vivo", "Tablero de Monitoreo (Capa Gold)"])

with tab1:
    st.subheader("Simulador de Transacción")
    with st.form("tx_form"):
        col1, col2 = st.columns(2)
        with col1:
            step = st.number_input("Hora acumulada (step 1-744)", min_value=1, max_value=744, value=10)
            tx_type = st.selectbox("Tipo de operación", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
            amount = st.number_input("Monto ($)", min_value=0.01, value=150000.0)
            is_merchant = st.selectbox("¿Destino es comercio ('M')?", [0, 1])
        with col2:
            old_org = st.number_input("Saldo inicial origen", min_value=0.0, value=150000.0)
            new_org = st.number_input("Saldo final origen", min_value=0.0, value=0.0)
            old_dest = st.number_input("Saldo inicial destino", min_value=0.0, value=0.0)
            new_dest = st.number_input("Saldo final destino", min_value=0.0, value=0.0)

        submitted = st.form_submit_button("Evaluar Transacción")

    if submitted:
        payload = {
            "step": int(step),
            "type": tx_type,
            "amount": float(amount),
            "oldbalanceOrg": float(old_org),
            "newbalanceOrig": float(new_org),
            "oldbalanceDest": float(old_dest),
            "newbalanceDest": float(new_dest),
            "is_merchant_dest": int(is_merchant)
        }
        try:
            response = requests.post(api_url, json=payload, timeout=8)
            response.raise_for_status()
            data = response.json()

            if "is_fraud" not in data or "fraud_probability" not in data:
                st.error("Respuesta inválida del servidor: faltan campos obligatorios.")
            elif data["is_fraud"] == 1:
                st.error(f"ALERTA DE FRAUDE ({data['decision']}) — Probabilidad: {data['fraud_probability']*100:.2f}%")
            else:
                st.success(f"TRANSACCIÓN SEGURA ({data['decision']}) — Probabilidad de fraude: {data['fraud_probability']*100:.2f}%")

        except requests.exceptions.Timeout:
            st.warning("Tiempo de espera agotado (Timeout): la API tardó demasiado en responder.")
        except requests.exceptions.ConnectionError:
            st.error("Error de conexión: no se pudo contactar al servicio del modelo publicado.")
        except requests.exceptions.HTTPError:
            st.error(f"Error HTTP {response.status_code}: datos rechazados por validación ({response.text}).")
        except ValueError:
            st.error("Respuesta inválida: el servidor no devolvió un JSON válido.")

with tab2:
    st.subheader("Indicadores Agregados sobre >4M de Transacciones (Capa Gold)")
    try:
        kpis = pd.read_csv("frontend_app/gold_kpis.csv") if False else pd.read_csv("gold_kpis.csv")
        c1, c2, c3 = st.columns(3)
        c1.metric("Transacciones Analizadas", f"{int(kpis['total_transacciones'].sum()):,}")
        c2.metric("Fraudes Detectados", f"{int(kpis['total_fraudes'].sum()):,}")
        c3.metric("Monto Total en Riesgo", f"${kpis['monto_en_riesgo'].sum():,.2f}")

        fig_type = px.bar(
            kpis.groupby("type", as_index=False)["total_fraudes"].sum(),
            x="type", y="total_fraudes", title="Fraudes por Tipo de Transacción", color="type"
        )
        st.plotly_chart(fig_type, use_container_width=True)

        fig_hour = px.line(
            kpis.groupby("hour_of_day", as_index=False)["tasa_fraude_pct"].mean(),
            x="hour_of_day", y="tasa_fraude_pct", title="Tasa Promedio de Fraude (%) por Hora del Día"
        )
        st.plotly_chart(fig_hour, use_container_width=True)
    except FileNotFoundError:
        st.info("Coloca el archivo gold_kpis.csv exportado de Databricks en la carpeta frontend_app/ para visualizar las gráficas.")
