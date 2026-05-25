"""
salvar_modelo.py
================
Execute este script UMA VEZ para gerar o arquivo 'modelo_obesidade.pkl'.
Esse arquivo será carregado pelo app.py do Streamlit.

Como rodar:
    python salvar_modelo.py
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# ── 1. Leitura dos dados ──────────────────────────────────────────────────────
URL_DADOS = (
    "https://raw.githubusercontent.com/jnssena/tech_challenge_4"
    "/main/Referencias_Atividade/Obesity.csv"
)

df = pd.read_csv(URL_DADOS)
print(f"Dados carregados: {df.shape[0]} linhas × {df.shape[1]} colunas")

COLUNA_ALVO = "Obesity"

# ── 2. Label Encoding ─────────────────────────────────────────────────────────
df_modelo = df.copy()

colunas_categoricas = df_modelo.select_dtypes(include="object").columns.tolist()
colunas_features_cat = [c for c in colunas_categoricas if c != COLUNA_ALVO]

encoders = {}

for col in colunas_features_cat:
    le = LabelEncoder()
    df_modelo[col] = le.fit_transform(df_modelo[col].astype(str))
    encoders[col] = le

le_target = LabelEncoder()
df_modelo[COLUNA_ALVO] = le_target.fit_transform(df_modelo[COLUNA_ALVO].astype(str))
encoders[COLUNA_ALVO] = le_target

print(f"\nClasses do target:")
for i, classe in enumerate(le_target.classes_):
    print(f"  {i} → {classe}")

# ── 3. Separação treino/teste ─────────────────────────────────────────────────
X = df_modelo.drop(columns=[COLUNA_ALVO])
y = df_modelo[COLUNA_ALVO]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── 4. Treinar e selecionar o melhor modelo ───────────────────────────────────
modelos = {
    "Regressão Logística": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Árvore de Decisão": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", DecisionTreeClassifier(random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
    "KNN (K-Vizinhos)": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", KNeighborsClassifier(n_neighbors=5))
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = {}

print("\nAvaliando modelos...")
for nome, pipeline in modelos.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    resultados[nome] = {"media": scores.mean(), "std": scores.std()}
    print(f"  {nome:25s} → {scores.mean():.4f} ± {scores.std():.4f}")

nome_melhor = max(resultados, key=lambda n: resultados[n]["media"])
modelo_final = modelos[nome_melhor]
modelo_final.fit(X_train, y_train)

y_pred = modelo_final.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")

print(f"\n✅ Melhor modelo: {nome_melhor}")
print(f"   Acurácia no teste : {acc:.4f} ({acc*100:.1f}%)")
print(f"   F1-Score          : {f1:.4f} ({f1*100:.1f}%)")

# ── 5. Salvar tudo num único arquivo .pkl ─────────────────────────────────────
pacote = {
    "modelo":        modelo_final,        # Pipeline treinado (scaler + modelo)
    "encoders":      encoders,            # LabelEncoders de todas as colunas
    "features":      list(X.columns),     # Nomes das colunas de entrada
    "nome_modelo":   nome_melhor,         # Nome do melhor modelo
    "acuracia":      acc,
    "f1_score":      f1,
}

joblib.dump(pacote, "modelo_obesidade.pkl")
print("\n✅ Arquivo 'modelo_obesidade.pkl' salvo com sucesso!")
print("   Agora você pode rodar o app.py no Streamlit.")
