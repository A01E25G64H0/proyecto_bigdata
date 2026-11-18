from pyspark.sql import functions as F

# Ajusta la ruta al archivo CSV subido en tu Databricks (Volume o DBFS)
RAW_CSV_PATH = "/Volumes/workspace/default/raw_data/onlinefraud.csv"

df_raw = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(RAW_CSV_PATH)
)

# Columnas de auditoría exigidas en Bronze
df_bronze = (
    df_raw
    .withColumn("ingestion_timestamp", F.current_timestamp())
    .withColumn("source_file", F.lit(RAW_CSV_PATH))
)

df_bronze.write.format("delta").mode("overwrite").saveAsTable("default.bronze_onlinefraud")

print(f"Total registros en Bronze: {spark.table('default.bronze_onlinefraud').count():,}")
