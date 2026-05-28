import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)

import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# CONFIGURAÇÃO DA PÁGINA
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sistema de Diagnóstico Hospitalar",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
# CSS CUSTOMIZADO — visual médico clean e profissional
# ══════════════════════════════════════════════════════
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');


html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.material-icons {
    font-family: 'Material Icons' !important;
}

/* ── Fundo geral: cinza muito suave ── */
.stApp {
    background: #f2f2f2;
}

/* ── Sidebar: tom marrom escuro #2c1810 ── */
[data-testid="stSidebar"] {
    background: #2c1810 !important;
    border-right: 1px solid #3d2318;
}
[data-testid="stSidebar"] * {
    color: #f0e6df !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #c9a898 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0em;
    text-transform: none;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Cards brancos ── */
.card {
    background: white;
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* ── Card header: gradiente marrom → marrom médio ── */
.card-blue {
    background: linear-gradient(135deg, #2c1810 0%, #5a3020 100%);
    color: white;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.card-blue h2, .card-blue p, .card-blue span {
    color: white !important;
}

/* ── Cards de resultado do diagnóstico ── */
.result-normal    { background: #e8f5e9; border: 2px solid #4caf50; }
.result-overweight{ background: #fff8e1; border: 2px solid #ffb300; }
.result-obese     { background: #fce4ec; border: 2px solid #e91e63; }
.result-insuf     { background: #e3f2fd; border: 2px solid #1e88e5; }

/* ── KPI cards do painel analítico ── */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.metric-value {
    font-family: 'Inter', sans-serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #2c1810;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
}

/* ── Título principal do header ── */
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: white;
    margin: 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 0.9rem;
    color: #c9a898;
    margin-top: 6px;
}

/* ── Abas: cor ativa alinhada ao tema marrom ── */
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    color: #64748b;
    border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    color: #2c1810 !important;
    border-bottom: 2px solid #2c1810 !important;
}

/* ── Botão: marrom coerente com a sidebar ── */
.stButton > button {
    background: linear-gradient(135deg, #2c1810, #5a3020) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100%;
    transition: all 0.2s;
    letter-spacing: 0.03em;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(44,24,16,0.4) !important;
}

/* Divisor */
.divider { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }

/* Oculta elementos padrão do streamlit */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# FUNÇÕES DE CACHE — treina o modelo só uma vez
# ══════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def carregar_dados():
    URL = (
        "https://raw.githubusercontent.com/jnssena/tech_challenge_4"
        "/main/Referencias_Atividade/Obesity.csv"
    )
    df = pd.read_csv(URL)
    return df

@st.cache_resource(show_spinner=False)
def treinar_modelo(df_hash):
    df = carregar_dados()
    df_modelo = df.copy()
    COLUNA_ALVO = "Obesity"

    colunas_cat = df_modelo.select_dtypes(include="object").columns.tolist()
    colunas_features_cat = [c for c in colunas_cat if c != COLUNA_ALVO]

    encoders = {}
    for col in colunas_features_cat:
        le = LabelEncoder()
        df_modelo[col] = le.fit_transform(df_modelo[col].astype(str))
        encoders[col] = le

    le_target = LabelEncoder()
    df_modelo[COLUNA_ALVO] = le_target.fit_transform(df_modelo[COLUNA_ALVO].astype(str))

    X = df_modelo.drop(columns=[COLUNA_ALVO])
    y = df_modelo[COLUNA_ALVO]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    modelo_final = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    modelo_final.fit(X_train, y_train)

    y_pred = modelo_final.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=le_target.classes_,
                                   output_dict=True)

    # Importância das features (Random Forest / GB)
    estimador = modelo_final.named_steps["modelo"]
    if hasattr(estimador, "feature_importances_"):
        feat_imp = pd.DataFrame({
            "Feature": X.columns,
            "Importância": estimador.feature_importances_
        }).sort_values("Importância", ascending=False)
    else:
        coefs = np.abs(estimador.coef_).mean(axis=0)
        feat_imp = pd.DataFrame({
            "Feature": X.columns,
            "Importância": coefs
        }).sort_values("Importância", ascending=False)

    return {
        "modelo": modelo_final,
        "encoders": encoders,
        "le_target": le_target,
        "colunas_features_cat": colunas_features_cat,
        "X_columns": list(X.columns),
        "acc": acc,
        "f1": f1,
        "cm": cm,
        "report": report,
        "feat_imp": feat_imp,
    }


# ══════════════════════════════════════════════════════
# CARREGAR DADOS E MODELO
# ══════════════════════════════════════════════════════
with st.spinner("🔄 Carregando dados e treinando o modelo..."):
    df = carregar_dados()
    art = treinar_modelo(hash(df.shape[0]))

modelo        = art["modelo"]
encoders      = art["encoders"]
le_target     = art["le_target"]
feat_cat      = art["colunas_features_cat"]
X_cols        = art["X_columns"]


# ══════════════════════════════════════════════════════
# SIDEBAR — formulário de entrada do paciente
# ══════════════════════════════════════════════════════
with st.sidebar:

    st.markdown("### Dados do Paciente")

    with st.expander("Dados Físicos", expanded=True):
        genero_pt = st.selectbox("Gênero", ["Masculino", "Feminino"])
        genero = {"Masculino": "Male", "Feminino": "Female"}[genero_pt]

        idade  = st.slider("Idade", 10, 80, 30)
        altura = st.slider("Altura (m)", 1.40, 2.10, 1.70, 0.01)
        peso   = st.slider("Peso (kg)", 30.0, 180.0, 75.0, 0.5)

    with st.expander("Hábitos Alimentares", expanded=True):
        hist_fam_pt = st.selectbox("Histórico familiar de sobrepeso", ["Sim", "Não"])
        hist_fam = {"Sim": "yes", "Não": "no"}[hist_fam_pt]

        favc_pt = st.selectbox("Consome alimentos calóricos frequentemente?", ["Sim", "Não"])
        favc = {"Sim": "yes", "Não": "no"}[favc_pt]

        fcvc = st.slider("Freq. consumo de vegetais (1=baixo, 3=alto)", 1.0, 3.0, 2.0, 0.5)
        ncp  = st.slider("Nº de refeições principais por dia", 1.0, 4.0, 3.0, 0.5)

        caec_pt = st.selectbox("Come entre as refeições?",
                               ["Não", "Às vezes", "Frequentemente", "Sempre"])
        caec = {"Não": "no", "Às vezes": "Sometimes",
                "Frequentemente": "Frequently", "Sempre": "Always"}[caec_pt]

        ch2o = st.slider("Litros de água por dia (1–3)", 1.0, 3.0, 2.0, 0.5)

        scc_pt = st.selectbox("Monitora calorias consumidas?", ["Não", "Sim"])
        scc = {"Sim": "yes", "Não": "no"}[scc_pt]

    with st.expander("Estilo de Vida", expanded=True):
        smoke_pt = st.selectbox("Fuma?", ["Não", "Sim"])
        smoke = {"Sim": "yes", "Não": "no"}[smoke_pt]

        faf = st.slider("Freq. atividade física (0=nunca, 3=sempre)", 0.0, 3.0, 1.0, 0.5)
        tue = st.slider("Horas/dia em telas (0–2)", 0.0, 2.0, 1.0, 0.5)

        calc_pt = st.selectbox("Freq. consumo de álcool",
                               ["Não consome", "Às vezes", "Frequentemente", "Sempre"])
        calc = {"Não consome": "no", "Às vezes": "Sometimes",
                "Frequentemente": "Frequently", "Sempre": "Always"}[calc_pt]

        mtrans_pt = st.selectbox("Principal meio de transporte",
                                 ["Transporte Público", "A pé", "Automóvel",
                                  "Moto", "Bicicleta"])
        mtrans = {
            "Transporte Público": "Public_Transportation",
            "A pé":               "Walking",
            "Automóvel":          "Automobile",
            "Moto":               "Motorbike",
            "Bicicleta":          "Bike",
        }[mtrans_pt]

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Relizar Diagnóstico")


# ══════════════════════════════════════════════════════
# HELPER — predição
# ══════════════════════════════════════════════════════
def fazer_predicao():
    dados = {
        "Gender": genero, "Age": float(idade), "Height": altura,
        "Weight": peso, "family_history": hist_fam, "FAVC": favc,
        "FCVC": fcvc, "NCP": ncp, "CAEC": caec, "SMOKE": smoke,
        "CH2O": ch2o, "SCC": scc, "FAF": faf, "TUE": tue,
        "CALC": calc, "MTRANS": mtrans,
    }
    pac = pd.DataFrame([dados])
    for col in feat_cat:
        le = encoders[col]
        val = pac[col].astype(str)
        val_safe = val.map(lambda v: v if v in le.classes_ else le.classes_[0])
        pac[col] = le.transform(val_safe)
    pac = pac[X_cols]
    pred_num  = modelo.predict(pac)[0]
    pred_nome = le_target.inverse_transform([pred_num])[0]
    probs     = modelo.predict_proba(pac)[0]
    return pred_nome, probs, le_target.classes_

def classe_css(nome):
    n = nome.lower()
    if "insufficient" in n: return "result-insuf"
    if "normal"       in n: return "result-normal"
    if "overweight"   in n: return "result-overweight"
    return "result-obese"

def emoji_classe(nome):
    n = nome.lower()
    if "insufficient" in n: return "🔵"
    if "normal"       in n: return "🟢"
    if "overweight"   in n: return "🟡"
    return "🔴"

CLASSE_PT = {
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight":       "Peso Normal",
    "Overweight_Level_I":  "Sobrepeso Grau I",
    "Overweight_Level_II": "Sobrepeso Grau II",
    "Obesity_Type_I":      "Obesidade Tipo I",
    "Obesity_Type_II":     "Obesidade Tipo II",
    "Obesity_Type_III":    "Obesidade Tipo III",
}

IMC = round(peso / (altura ** 2), 1)


# ══════════════════════════════════════════════════════
# HEADER HERO
# ══════════════════════════════════════════════════════
st.markdown(f"""
<div class="card-blue" style="padding:32px 36px; margin-bottom:24px">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px">
        <div>
            <h1 class="hero-title">Sistema de Diagnóstico Hospitalar</h1>
            <p class="hero-sub">Modelo de Machine Learning — <strong style="color:white">Gradient Boosting</strong></p>
        </div>
        <div style="display:flex; gap:24px; flex-wrap:wrap">
            <div style="text-align:center">
                <div style="font-family:'Inter',sans-serif; font-size:2rem; font-weight:600; color:white">{art['acc']*100:.1f}%</div>
                <div style="font-size:0.7rem; color:#c9a898; letter-spacing:0.1em">Acurácia</div>
            </div>
            <div style="text-align:center">
                <div style="font-family:'Inter',sans-serif; font-size:2rem; font-weight:600; color:white">{len(df)}</div>
                <div style="font-size:0.7rem; color:#c9a898; letter-spacing:0.1em">Pacientes</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# ABAS PRINCIPAIS
# ══════════════════════════════════════════════════════
tab1, tab2 = st.tabs([
    "Diagnóstico",
    "Painel Analítico",
])


# ══════════════════════════════════════════════════════
# ABA 1 — DIAGNÓSTICO
# ══════════════════════════════════════════════════════
with tab1:
    if predict_btn or "pred_nome" in st.session_state:
        if predict_btn:
            pred_nome, probs, classes = fazer_predicao()
            st.session_state["pred_nome"] = pred_nome
            st.session_state["probs"]     = probs
            st.session_state["classes"]   = classes
        else:
            pred_nome = st.session_state["pred_nome"]
            probs     = st.session_state["probs"]
            classes   = st.session_state["classes"]

        pred_pt  = CLASSE_PT.get(pred_nome, pred_nome)
        css_cls  = classe_css(pred_nome)
        emoji    = emoji_classe(pred_nome)
        confianca= probs.max() * 100

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown(f"""
            <div class="card {css_cls}" style="text-align:center; padding:36px">
                <div style="font-size:0.75rem; letter-spacing:0.12em; color:#888; margin-bottom:6px">Diagnóstico:</div>
                <div style="font-family:'Inter',sans-serif; font-size:2rem; font-weight:600; color:#2c1810; line-height:1.2">{pred_pt}</div>
                <div style="font-size:0.85rem; color:#555; margin-top:8px">Confiança do modelo: <strong>{confianca:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            # IMC calculado
            imc_class = (
                "Abaixo do Peso" if IMC < 18.5 else
                "Peso Normal"    if IMC < 25   else
                "Sobrepeso"      if IMC < 30   else
                "Obesidade"
            )

            st.markdown(f"""
            <div class="card" style="display:flex; justify-content:space-around; align-items:center; padding:20px 28px">
                <div style="text-align:center">
                    <div class="metric-value">{IMC}</div>
                    <div class="metric-label">IMC Calculado</div>
                </div>
                <div style="width:1px; background:#e2e8f0; height:50px"></div>
                <div style="text-align:center">
                    <div style="font-size:1rem; font-weight:600; color:#2c1810">{imc_class}</div>
                    <div class="metric-label">Classificação OMS</div>
                </div>
                <div style="width:1px; background:#e2e8f0; height:50px"></div>
                <div style="text-align:center">
                    <div class="metric-value">{peso:.0f}</div>
                    <div class="metric-label">Peso (kg)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Probabilidade por Classe:**")

            prob_df = pd.DataFrame({
                "Classe": [CLASSE_PT.get(c, c) for c in classes],
                "Prob":   probs * 100
            }).sort_values("Prob", ascending=True)

            COR_MAP = {
                "Abaixo do Peso":    "#1e88e5",
                "Peso Normal":       "#43a047",
                "Sobrepeso Grau I":  "#fb8c00",
                "Sobrepeso Grau II": "#f4511e",
                "Obesidade Tipo I":  "#e53935",
                "Obesidade Tipo II": "#c62828",
                "Obesidade Tipo III":"#880e4f",
            }
            cores_barras = [
                COR_MAP.get(c, "#64748b") for c in prob_df["Classe"]
            ]

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            bars = ax.barh(prob_df["Classe"], prob_df["Prob"],
                           color=cores_barras, edgecolor="white",
                           height=0.55, alpha=0.88)
            for bar, val in zip(bars, prob_df["Prob"]):
                ax.text(min(val + 1.5, 101), bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=9,
                        color="#333", fontweight="bold")
            ax.set_xlim(0, 115)
            ax.set_xlabel("Probabilidade (%)", fontsize=9, color="#555")
            ax.tick_params(axis="both", labelsize=9, colors="#555")
            ax.spines[["top","right","bottom"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.xaxis.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:60px 40px; border: 2px dashed #cbd5e1">
            <div style="color:#64748b; font-size:0.95rem">
                Preencha os dados do paciente na barra lateral<br>
                e clique em <strong>Relizar Diagnóstico</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# ABA 2 — PAINEL ANALÍTICO
# ══════════════════════════════════════════════════════
with tab2:

    # Paleta de marrons coerente com o tema #2c1810
    TEMA = {
        "escuro":    "#2c1810",
        "medio":     "#5a3020",
        "claro":     "#8b5e3c",
        "suave":     "#c9a898",
        "muitoClaro":"#f0e6df",
        "acento":    "#a0522d",
    }

    # Gradiente de 7 tons do mais claro ao mais escuro para as classes de obesidade
    CORES_CLASSES_TEMA = [
        "#f0e6df",  # Abaixo do Peso   — tom mais claro
        "#c9a898",  # Peso Normal
        "#a0805c",  # Sobrepeso I
        "#8b5e3c",  # Sobrepeso II
        "#5a3020",  # Obesidade I
        "#3d1e10",  # Obesidade II
        "#2c1810",  # Obesidade III    — tom mais escuro
    ]

    order = ["Insufficient_Weight","Normal_Weight",
             "Overweight_Level_I","Overweight_Level_II",
             "Obesity_Type_I","Obesity_Type_II","Obesity_Type_III"]

    # ── KPIs: Obesidade | Sobrepeso | IMC médio | Hist. Familiar
    obesos_pct    = df["Obesity"].str.contains("Obesity").mean() * 100
    sobrepeso_pct = df["Obesity"].str.contains("Overweight").mean() * 100
    imc_medio     = (df["Weight"] / (df["Height"] ** 2)).mean()
    hist_pct      = (df["family_history"] == "yes").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, fmt in [
        (c1, obesos_pct, "Com Obesidade", "{:.0f}%"),
        (c2, sobrepeso_pct, "Com Sobrepeso", "{:.0f}%"),
        (c3, imc_medio, "IMC Médio", "{:.1f}"),
        (c4, hist_pct, "Hist. Familiar", "{:.0f}%"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{fmt.format(val)}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Linha 1: Distribuição por Gênero (barras verticais) + Boxplot de Peso
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Distribuição por Nível de Obesidade e Gênero**")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        # Barras verticais agrupadas — hue=Gender, x=Obesity
        pal_genero = {"Male": TEMA["escuro"], "Female": TEMA["claro"]}
        data_plot = df[df["Obesity"].isin(order)].copy()
        data_plot["Obesity_PT"] = data_plot["Obesity"].map(CLASSE_PT)
        order_pt = [CLASSE_PT[o] for o in order]
        sns.countplot(
            data=data_plot, x="Obesity_PT", hue="Gender",
            order=order_pt, palette=pal_genero, ax=ax, edgecolor="white", width=0.7
        )
        ax.set_xlabel("")
        ax.set_ylabel("Nº de Pacientes", fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7.5)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["Masculino", "Feminino"],
                  title="Gênero", fontsize=8, title_fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Gráfico Boxplot de Peso por Classe de Obesidade**")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for i, classe in enumerate(order):
            dados_c = df[df["Obesity"] == classe]["Weight"]
            ax.boxplot(
                dados_c,
                positions=[i],
                patch_artist=True,
                boxprops=dict(facecolor=CORES_CLASSES_TEMA[i], alpha=0.85),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color="#aaa"),
                capprops=dict(color="#aaa"),
                flierprops=dict(marker="o", markerfacecolor="#bbb",
                                markersize=2, alpha=0.4),
                widths=0.55
            )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([CLASSE_PT.get(c, c) for c in order],
                           rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Peso (kg)", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Linha 2: Consumo de vegetais + Atividade física
    col_c, col_d = st.columns(2, gap="large")

    with col_c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Consumo de Vegetais por Nível de Obesidade**")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        fcvc_media = df.groupby("Obesity")["FCVC"].mean().reindex(order)
        fcvc_media.index = [CLASSE_PT.get(i, i) for i in fcvc_media.index]
        bars = ax.bar(
            fcvc_media.index, fcvc_media.values,
            color=CORES_CLASSES_TEMA, edgecolor="white", alpha=0.9, width=0.6
        )
        ax.axhline(fcvc_media.mean(), color=TEMA["acento"], linestyle="--",
                   alpha=0.7, linewidth=1.2, label="Média geral")
        for bar, val in zip(bars, fcvc_media.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#333")
        ax.set_ylabel("Freq. Consumo de Vegetais (1–3)", fontsize=9)
        ax.set_xticklabels(fcvc_media.index, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylim(0, fcvc_media.max() * 1.18)
        ax.spines[["top","right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_d:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Atividade Física Média por Classe**")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        faf_media = df.groupby("Obesity")["FAF"].mean().reindex(order)
        faf_media.index = [CLASSE_PT.get(i, i) for i in faf_media.index]
        cores_faf = [TEMA["escuro"] if v >= faf_media.mean() else TEMA["suave"]
                     for v in faf_media.values]
        bars = ax.barh(faf_media.index, faf_media.values,
                       color=cores_faf, edgecolor="white", alpha=0.9, height=0.55)
        ax.axvline(faf_media.mean(), color=TEMA["acento"], linestyle="--",
                   alpha=0.7, linewidth=1.2, label="Média geral")
        for bar, val in zip(bars, faf_media.values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=8, color="#333")
        ax.set_xlabel("Freq. de Atividade Física (0–3)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top","right"]].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()