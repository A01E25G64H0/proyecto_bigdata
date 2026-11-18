import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

# 1. Lectura desde Gold (conservando el 100% de fraudes y una muestra representativa de legítimas)
df_gold_spark = spark.read.table("default.gold_onlinefraud")
frauds_df = df_gold_spark.filter(F.col("isFraud") == 1).toPandas()
legit_df = df_gold_spark.filter(F.col("isFraud") == 0).sample(withReplacement=False, fraction=0.08, seed=42).toPandas()

df = pd.concat([frauds_df, legit_df], ignore_index=True)

feature_cols = [
    "type", "hour_of_day", "log_amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "is_merchant_dest", "errorBalanceOrig", "errorBalanceDest"
]
X = df[feature_cols]
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Preprocesador integrado (Codificación One-Hot + Escalado)
cat_cols = ["type"]
num_cols = [c for c in feature_cols if c != "type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

models = {
    "LogisticRegression_Balanced": LogisticRegression(class_weight="balanced", max_iter=500, random_state=42),
    "RandomForest_Balanced": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42)
}

mlflow.set_experiment("/Shared/Fraude_Financiero_PaySim")

best_f1 = -1
best_pipeline = None

for name, clf in models.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_prob)
        
        mlflow.log_param("model_type", name)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        
        print(f"{name} -> F1: {f1:.4f} | PR-AUC: {pr_auc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipe

# Exportar el mejor modelo completo (preprocesador + clasificador)
joblib.dump(best_pipeline, "/tmp/fraud_model.pkl")
print("Modelo ganador guardado en /tmp/fraud_model.pkl")

import shutil
shutil.copy("/tmp/fraud_model.pkl", "/Volumes/workspace/default/raw_data/fraud_model.pkl")
