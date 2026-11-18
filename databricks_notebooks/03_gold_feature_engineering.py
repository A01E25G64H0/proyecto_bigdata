from pyspark.sql import functions as F

df_silver = spark.read.table("default.silver_onlinefraud")

# 1. Ingeniería de características de dominio
df_gold = (
    df_silver
    .withColumn("hour_of_day", F.col("step") % 24)
    .withColumn("errorBalanceOrig", F.col("oldbalanceOrg") - F.col("amount") - F.col("newbalanceOrig"))
    .withColumn("errorBalanceDest", F.col("oldbalanceDest") + F.col("amount") - F.col("newbalanceDest"))
    .select(
        "type", "hour_of_day", "log_amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "is_merchant_dest", "errorBalanceOrig", "errorBalanceDest",
        "isFraud"
    )
)

df_gold.write.format("delta").mode("overwrite").saveAsTable("default.gold_onlinefraud")

# 2. Tabla agregada para BI / Dashboard en Streamlit
df_kpis = (
    df_silver
    .withColumn("hour_of_day", F.col("step") % 24)
    .groupBy("hour_of_day", "type")
    .agg(
        F.count("*").alias("total_transacciones"),
        F.sum("isFraud").alias("total_fraudes"),
        F.round(F.avg("amount"), 2).alias("monto_promedio"),
        F.round(F.sum(F.when(F.col("isFraud") == 1, F.col("amount")).otherwise(0)), 2).alias("monto_en_riesgo")
    )
    .withColumn("tasa_fraude_pct", F.round((F.col("total_fraudes") / F.col("total_transacciones")) * 100, 4))
    .orderBy("hour_of_day", "type")
)

df_kpis.write.format("delta").mode("overwrite").saveAsTable("default.gold_kpis")

# Exporta este resumen pequeño a Pandas para descargarlo como gold_kpis.csv hacia Streamlit
display(df_kpis)
