"""
app.py — Aplicação Streamlit para Predição de Obesidade
========================================================
Como rodar localmente:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🩺",
    layout="centered",
)

# ── Carregar o modelo salvo ───────────────────────────────────────────────────
@st.cache_resource          # Carrega o modelo apenas uma vez (fica na memória)
def carregar_modelo():
    return joblib.load("modelo_obesidade.pkl")

pacote = carregar_modelo()
modelo   = pacote["modelo"]
encoders = pacote["encoders"]
features = pacote["features"]

# Encoder do target (para transformar número de volta em nome)
le_target = encoders["Obesity"]

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.title("🩺 Sistema Preditivo de Obesidade")
st.markdown(
    """
    Preencha as informações do paciente nos campos abaixo e clique em
    **Prever** para obter o diagnóstico de nível de obesidade.
    """
)

st.divider()

# ── Formulário de entrada ─────────────────────────────────────────────────────
st.subheader("📋 Dados do Paciente")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Sexo",
        options=["Female", "Male"],
        help="Sexo biológico do paciente"
    )

    age = st.number_input(
        "Idade (anos)",
        min_value=1, max_value=120, value=25, step=1
    )

    height = st.number_input(
        "Altura (metros)",
        min_value=0.50, max_value=2.50, value=1.70, step=0.01, format="%.2f"
    )

    weight = st.number_input(
        "Peso (kg)",
        min_value=10.0, max_value=300.0, value=70.0, step=0.5, format="%.1f"
    )

    family_history = st.selectbox(
        "Histórico familiar de sobrepeso?",
        options=["yes", "no"],
        help="Algum familiar tem ou teve sobrepeso?"
    )

    favc = st.selectbox(
        "Consome alimentos calóricos com frequência?",
        options=["yes", "no"],
        help="FAVC — Frequent Consumption of High Caloric Food"
    )

    fcvc = st.slider(
        "Frequência de consumo de vegetais (0–3)",
        min_value=1.0, max_value=3.0, value=2.0, step=0.5,
        help="FCVC: 1 = raramente, 3 = sempre"
    )

    ncp = st.slider(
        "Número de refeições principais por dia",
        min_value=1.0, max_value=4.0, value=3.0, step=0.5,
        help="NCP: Number of Main Meals"
    )

with col2:
    caec = st.selectbox(
        "Come entre as refeições?",
        options=["no", "Sometimes", "Frequently", "Always"],
        help="CAEC — Consumption of Food Between Meals"
    )

    smoke = st.selectbox(
        "Fumante?",
        options=["no", "yes"]
    )

    ch2o = st.slider(
        "Consumo diário de água (litros, 1–3)",
        min_value=1.0, max_value=3.0, value=2.0, step=0.5,
        help="CH2O: Daily Water Consumption"
    )

    scc = st.selectbox(
        "Monitora as calorias ingeridas?",
        options=["no", "yes"],
        help="SCC — Caloric Intake Monitoring"
    )

    faf = st.slider(
        "Frequência de atividade física por semana (0–3)",
        min_value=0.0, max_value=3.0, value=1.0, step=0.5,
        help="FAF: Physical Activity Frequency"
    )

    tue = st.slider(
        "Horas diárias usando dispositivos tecnológicos (0–2)",
        min_value=0.0, max_value=2.0, value=1.0, step=0.5,
        help="TUE: Time Using Technology Devices"
    )

    calc = st.selectbox(
        "Frequência de consumo de álcool",
        options=["no", "Sometimes", "Frequently", "Always"],
        help="CALC — Consumption of Alcohol"
    )

    mtrans = st.selectbox(
        "Principal meio de transporte",
        options=["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
        help="MTRANS — Transportation Used"
    )

st.divider()

# ── Botão de predição ─────────────────────────────────────────────────────────
if st.button("🔍 Prever Nível de Obesidade", use_container_width=True, type="primary"):

    # Monta o dicionário com os valores inseridos
    dados_brutos = {
        "Gender":                          gender,
        "Age":                             age,
        "Height":                          height,
        "Weight":                          weight,
        "family_history_with_overweight":  family_history,
        "FAVC":                            favc,
        "FCVC":                            fcvc,
        "NCP":                             ncp,
        "CAEC":                            caec,
        "SMOKE":                           smoke,
        "CH2O":                            ch2o,
        "SCC":                             scc,
        "FAF":                             faf,
        "TUE":                             tue,
        "CALC":                            calc,
        "MTRANS":                          mtrans,
    }

    # Cria DataFrame com uma linha
    df_entrada = pd.DataFrame([dados_brutos])

    # Aplica o LabelEncoder nas colunas categóricas (mesmo encoding do treino)
    for col in df_entrada.columns:
        if col in encoders and col != "Obesity":
            le = encoders[col]
            valor = df_entrada[col].astype(str)
            # Se aparecer valor desconhecido, usa o primeiro valor conhecido
            df_entrada[col] = valor.apply(
                lambda v: le.transform([v])[0]
                if v in le.classes_
                else le.transform([le.classes_[0]])[0]
            )

    # Garante a ordem correta das colunas
    df_entrada = df_entrada[features]

    # Faz a predição
    predicao_num = modelo.predict(df_entrada)[0]
    predicao_proba = modelo.predict_proba(df_entrada)[0]
    predicao_nome = le_target.inverse_transform([predicao_num])[0]

    # ── Exibição do resultado ──────────────────────────────────────────────────
    st.subheader("📊 Resultado da Predição")

    # Mapeamento de cores e ícones por nível de obesidade
    mapa_resultado = {
        "Insufficient_Weight":       ("🔵", "Abaixo do Peso",           "#1E90FF"),
        "Normal_Weight":             ("🟢", "Peso Normal",               "#28A745"),
        "Overweight_Level_I":        ("🟡", "Sobrepeso Grau I",          "#FFC107"),
        "Overweight_Level_II":       ("🟠", "Sobrepeso Grau II",         "#FF8C00"),
        "Obesity_Type_I":            ("🔴", "Obesidade Tipo I",          "#DC3545"),
        "Obesity_Type_II":           ("🟣", "Obesidade Tipo II",         "#9B59B6"),
        "Obesity_Type_III":          ("⚫", "Obesidade Tipo III (Grave)", "#343A40"),
    }

    icone, descricao, cor = mapa_resultado.get(
        predicao_nome, ("⚪", predicao_nome, "#6C757D")
    )

    st.markdown(
        f"""
        <div style="
            background-color: {cor}22;
            border-left: 6px solid {cor};
            padding: 20px 24px;
            border-radius: 8px;
            margin-bottom: 16px;
        ">
            <h2 style="color:{cor}; margin:0;">{icone} {descricao}</h2>
            <p style="color:#555; margin:4px 0 0 0; font-size:13px;">
                Classificação bruta: <code>{predicao_nome}</code>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probabilidades por classe
    st.markdown("**Probabilidade por classe:**")
    classes_nomes = le_target.inverse_transform(
        np.arange(len(le_target.classes_))
    )
    df_proba = pd.DataFrame({
        "Nível de Obesidade": classes_nomes,
        "Probabilidade (%)": (predicao_proba * 100).round(2),
    }).sort_values("Probabilidade (%)", ascending=False)

    st.dataframe(df_proba, use_container_width=True, hide_index=True)

    # IMC calculado
    imc = weight / (height ** 2)
    st.info(f"📐 IMC calculado: **{imc:.2f}** kg/m²")

# ── Rodapé ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Modelo utilizado: **{pacote['nome_modelo']}** | "
    f"Acurácia: **{pacote['acuracia']*100:.1f}%** | "
    f"F1-Score: **{pacote['f1_score']*100:.1f}%**"
)
st.caption("⚠️ Este sistema é uma ferramenta de apoio à decisão. O diagnóstico final deve ser realizado por um profissional de saúde.")
