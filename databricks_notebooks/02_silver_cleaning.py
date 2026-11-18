from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

df_bronze = spark.read.table("default.bronze_onlinefraud")

# 1. Tipos estrictos y eliminación de duplicados exactos
df_typed = (
    df_bronze
    .dropDuplicates(["step", "type", "amount", "nameOrig", "nameDest"])
    .withColumn("step", F.col("step").cast(IntegerType()))
    .withColumn("amount", F.col("amount").cast(DoubleType()))
    .withColumn("oldbalanceOrg", F.col("oldbalanceOrg").cast(DoubleType()))
    .withColumn("newbalanceOrig", F.col("newbalanceOrig").cast(DoubleType()))
    .withColumn("oldbalanceDest", F.col("oldbalanceDest").cast(DoubleType()))
    .withColumn("newbalanceDest", F.col("newbalanceDest").cast(DoubleType()))
    .withColumn("isFraud", F.col("isFraud").cast(IntegerType()))
)

# 2. Nulos encubiertos en comercios ('M...') y borrado de columnas sin varianza o IDs
df_nulls = (
    df_typed
    .dropna(subset=["type", "amount", "isFraud"])
    .withColumn("is_merchant_dest", F.when(F.col("nameDest").startswith("M"), 1).otherwise(0))
    .drop("nameOrig", "nameDest", "isFlaggedFraud", "ingestion_timestamp", "source_file")
)

# 3. Tratamiento de outliers: filtro > 0, Winsorización al percentil 99 y log1p
p99_amount = df_nulls.approxQuantile("amount", [0.99], 0.01)[0]

df_silver = (
    df_nulls
    .filter(F.col("amount") > 0)
    .withColumn("amount_capped", F.least(F.col("amount"), F.lit(p99_amount)))
    .withColumn("log_amount", F.log1p(F.col("amount_capped")))
)

df_silver.write.format("delta").mode("overwrite").saveAsTable("default.silver_onlinefraud")
print(f"Percentil 99 aplicado: {p99_amount:,.2f} | Registros en Silver: {df_silver.count():,}")
