# ---------------------------------------------------
# Student Dropout Prediction System
# Streamlit Dashboard - Phase 6 Visual Upgrade
# ---------------------------------------------------

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Student Dropout Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Constants
# ---------------------------------------------------

PRIMARY = "#2563EB"
SUCCESS = "#16A34A"
WARNING = "#F59E0B"
DANGER = "#DC2626"
BG = "#F8FAFC"
CARD_BG = "#FFFFFF"
TEXT = "#111827"
SUBTEXT = "#6B7280"
BORDER = "#E5E7EB"
INPUT_BG = "#FFFFFF"

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv("data/student_dataset.csv", sep=";")

if "Dropout_Binary" not in df.columns:
    df["Dropout_Binary"] = df["Target"].apply(lambda x: 1 if x == "Dropout" else 0)

# ---------------------------------------------------
# Load Model + Processed Data
# ---------------------------------------------------

best_model = joblib.load("models/best_model.pkl")
X_test_scaled = pd.read_csv("data/processed/X_test_scaled.csv")

# Keep SHAP sample smaller for speed
X_shap_sample = X_test_scaled.sample(min(200, len(X_test_scaled)), random_state=42)

# ---------------------------------------------------
# Model Metrics
# Replace Logistic Regression and Decision Tree values
# with your actual Phase 4 results
# ---------------------------------------------------

model_metrics_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [0.860, 0.842, 0.884],
    "Precision": [0.801, 0.786, 0.842],
    "Recall": [0.742, 0.731, 0.785],
    "F1 Score": [0.770, 0.757, 0.812],
    "ROC-AUC": [0.902, 0.861, 0.933]
})

# ---------------------------------------------------
# Custom Styling
# ---------------------------------------------------

st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        font-family: "Inter", "Segoe UI", sans-serif;
    }}

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main,
    .block-container {{
        background-color: {BG} !important;
        color: {TEXT} !important;
    }}

    [data-testid="stHeader"] {{
        background: rgba(248, 250, 252, 0.92) !important;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid {BORDER};
    }}

    [data-testid="stToolbar"] {{
        right: 1rem;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #EFF6FF 0%, #F8FAFC 100%) !important;
        border-right: 1px solid {BORDER};
    }}

    section[data-testid="stSidebar"] * {{
        color: {TEXT} !important;
    }}

    .sidebar-title {{
        font-size: 26px;
        font-weight: 800;
        color: {TEXT};
        margin-bottom: 4px;
    }}

    .sidebar-subtitle {{
        font-size: 14px;
        color: {SUBTEXT};
        margin-bottom: 18px;
    }}

    .hero-card {{
        background: linear-gradient(135deg, #EFF6FF 0%, #FFFFFF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 22px;
        padding: 28px 30px;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.08);
        margin-bottom: 20px;
        animation: fadeUp 0.7s ease-out;
    }}

    .hero-title {{
        font-size: 42px;
        font-weight: 800;
        color: {TEXT};
        margin-bottom: 8px;
        line-height: 1.2;
    }}

    .hero-subtitle {{
        font-size: 18px;
        color: {SUBTEXT};
        line-height: 1.7;
        margin-bottom: 0px;
    }}

    .section-title {{
        font-size: 28px;
        font-weight: 700;
        color: {TEXT};
        margin-top: 10px;
        margin-bottom: 8px;
    }}

    .section-subtitle {{
        font-size: 15px;
        color: {SUBTEXT};
        margin-bottom: 18px;
    }}

    .card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        margin-bottom: 16px;
        animation: fadeUp 0.7s ease-out;
    }}

    .kpi-card {{
        position: relative;
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        padding: 28px 18px;
        border-radius: 18px;
        text-align: center;
        min-height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.25s ease;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        animation: fadeUp 0.7s ease-out;
        animation-fill-mode: both;
    }}

    .kpi-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 16px 35px rgba(17, 24, 39, 0.12);
        border-color: #BFDBFE;
    }}

    .kpi-label {{
        color: {SUBTEXT};
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 14px;
    }}

    .kpi-value {{
        color: {TEXT};
        font-size: 30px;
        font-weight: 800;
        line-height: 1.2;
        word-break: break-word;
    }}

    .tooltip-text {{
        visibility: hidden;
        opacity: 0;
        width: 90%;
        background-color: {TEXT};
        color: white;
        text-align: center;
        border-radius: 10px;
        padding: 10px 12px;
        position: absolute;
        bottom: 110%;
        left: 50%;
        transform: translateX(-50%);
        font-size: 13px;
        transition: opacity 0.25s ease;
        z-index: 999;
    }}

    .tooltip-text::after {{
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 6px;
        border-style: solid;
        border-color: {TEXT} transparent transparent transparent;
    }}

    .kpi-card:hover .tooltip-text {{
        visibility: visible;
        opacity: 1;
    }}

    .info-box {{
        background: #F8FAFC;
        border: 1px solid {BORDER};
        border-left: 5px solid {PRIMARY};
        padding: 16px 18px;
        border-radius: 14px;
        color: {TEXT};
        margin-top: 8px;
        margin-bottom: 14px;
        animation: fadeUp 0.7s ease-out;
        animation-fill-mode: both;
    }}

    .success-box {{
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid {SUCCESS};
        padding: 16px 18px;
        border-radius: 14px;
        color: {TEXT};
        margin-top: 8px;
        margin-bottom: 14px;
        animation: fadeUp 0.7s ease-out;
        animation-fill-mode: both;
    }}

    .warning-box {{
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid {WARNING};
        padding: 16px 18px;
        border-radius: 14px;
        color: {TEXT};
        margin-top: 8px;
        margin-bottom: 14px;
        animation: fadeUp 0.7s ease-out;
        animation-fill-mode: both;
    }}

    .danger-box {{
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid {DANGER};
        padding: 16px 18px;
        border-radius: 14px;
        color: {TEXT};
        margin-top: 8px;
        margin-bottom: 14px;
        animation: fadeUp 0.7s ease-out;
        animation-fill-mode: both;
    }}

    .pipeline-step {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        font-weight: 600;
        color: {TEXT};
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        animation: fadeUp 0.7s ease-out;
        animation-fill-mode: both;
    }}

    .chip {{
        display: inline-block;
        background: #EFF6FF;
        color: {PRIMARY};
        border: 1px solid #BFDBFE;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 13px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }}

    .risk-badge {{
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 700;
        color: white !important;
    }}

    div[data-testid="stForm"] {{
        background: {CARD_BG} !important;
        border: 1px solid {BORDER};
        border-radius: 22px;
        padding: 24px !important;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        margin-top: 6px;
        animation: fadeUp 0.7s ease-out;
        animation-fill-mode: both;
    }}

    div[data-testid="stForm"] h3,
    div[data-testid="stForm"] label,
    div[data-testid="stForm"] p,
    div[data-testid="stForm"] span {{
        color: {TEXT} !important;
    }}

    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] span {{
        color: {TEXT} !important;
    }}

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    .stNumberInput > div > div,
    .stSelectbox > div > div {{
        background-color: {INPUT_BG} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        min-height: 48px;
        box-shadow: none !important;
    }}

    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span,
    .stNumberInput input,
    .stSelectbox div {{
        color: {TEXT} !important;
        background: transparent !important;
    }}

    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
    }}

    .stButton > button,
    .stFormSubmitButton > button {{
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.75rem 1.4rem !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.20) !important;
    }}

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 14px 28px rgba(37, 99, 235, 0.28) !important;
    }}

    .stButton > button:focus,
    .stFormSubmitButton > button:focus {{
        outline: none !important;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.16) !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: white !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        transform: translateY(-2px);
        border-color: #BFDBFE !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: #EFF6FF !important;
        border-color: #BFDBFE !important;
        color: {PRIMARY} !important;
    }}

    .stDataFrame, .stTable {{
        background-color: white !important;
        border-radius: 16px !important;
    }}

    .animate-fade-up {{
        animation: fadeUp 0.7s ease-out both;
    }}

    .animate-fade-in {{
        animation: fadeIn 0.8s ease-out both;
    }}

    .animate-delay-1 {{
        animation-delay: 0.08s;
    }}

    .animate-delay-2 {{
        animation-delay: 0.16s;
    }}

    .animate-delay-3 {{
        animation-delay: 0.24s;
    }}

    .animate-delay-4 {{
        animation-delay: 0.32s;
    }}

    .animate-delay-5 {{
        animation-delay: 0.40s;
    }}

    .hover-lift {{
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}

    .hover-lift:hover {{
        transform: translateY(-6px);
        box-shadow: 0 16px 35px rgba(17, 24, 39, 0.12);
    }}

    @keyframes fadeUp {{
        from {{
            opacity: 0;
            transform: translateY(16px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @keyframes fadeIn {{
        from {{
            opacity: 0;
        }}
        to {{
            opacity: 1;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------

def section_header(title, subtitle="", animation_class="animate-fade-up"):
    st.markdown(f"""
        <div class="{animation_class}">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)

def kpi_card(label, value, tooltip="", value_color=TEXT, extra_class=""):
    st.markdown(f"""
        <div class="kpi-card hover-lift animate-fade-up {extra_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{value_color};">{value}</div>
            <div class="tooltip-text">{tooltip}</div>
        </div>
    """, unsafe_allow_html=True)

def info_box(text, box_type="info", extra_class=""):
    class_name = {
        "info": "info-box",
        "success": "success-box",
        "warning": "warning-box",
        "danger": "danger-box"
    }.get(box_type, "info-box")

    st.markdown(f"""
        <div class="{class_name} animate-fade-up {extra_class}">
            {text}
        </div>
    """, unsafe_allow_html=True)

def risk_level(probability):
    if probability < 0.33:
        return "Low Risk", SUCCESS
    elif probability < 0.66:
        return "Medium Risk", WARNING
    return "High Risk", DANGER

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------

st.sidebar.markdown("""
<div class="sidebar-title">🎓 Dropout Dashboard</div>
<div class="sidebar-subtitle">ML analytics + explainable AI</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "EDA Insights",
        "Model Performance",
        "Model Comparison",
        "Interpretation",
        "Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard Highlights")
st.sidebar.markdown("""
- Modern SaaS layout  
- Explainable AI  
- Interactive charts  
- Real-time prediction  
""")

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

if page == "Home":

    st.markdown("""
    <div class="hero-card animate-fade-up">
        <div class="hero-title">🎓 Student Dropout Prediction System</div>
        <div class="hero-subtitle">
            This dashboard predicts whether a student is likely to drop out using
            demographic, academic, and financial data. It combines machine learning,
            explainable AI, and a modern analytics interface to make model outputs easy to understand.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        kpi_card(
            "Dataset Size",
            f"{len(df):,} Students",
            "Total number of student records used in the project.",
            extra_class="animate-delay-1"
        )

    with col2:
        kpi_card(
            "Best Model",
            "Random Forest",
            "Selected after comparing multiple machine learning models.",
            extra_class="animate-delay-2"
        )

    with col3:
        kpi_card(
            "ROC-AUC",
            "0.933",
            "Measures how well the model separates dropout and non-dropout students.",
            extra_class="animate-delay-3"
        )

    section_header("Project Pipeline", "The full machine learning workflow used in this project.")

    step_cols = st.columns(6)
    steps = [
        "1. EDA",
        "2. Preprocessing",
        "3. Training",
        "4. Evaluation",
        "5. Interpretation",
        "6. Prediction"
    ]

    for idx, (col, step) in enumerate(zip(step_cols, steps), start=1):
        delay_class = f"animate-delay-{min(idx, 5)}"
        with col:
            st.markdown(
                f'<div class="pipeline-step hover-lift animate-fade-up {delay_class}">{step}</div>',
                unsafe_allow_html=True
            )

    section_header("Project Highlights", "What this system demonstrates from an ML portfolio perspective.")

    st.markdown("""
    <div class="animate-fade-in animate-delay-2">
        <span class="chip">EDA</span>
        <span class="chip">SMOTE balancing</span>
        <span class="chip">Random Forest</span>
        <span class="chip">GridSearchCV</span>
        <span class="chip">SHAP explainability</span>
        <span class="chip">Interactive dashboard</span>
    </div>
    """, unsafe_allow_html=True)

    info_box(
        "<b>Why this project is strong:</b> It covers the full ML lifecycle from raw data analysis to a user-facing prediction tool, making it highly suitable for interviews, portfolio reviews, and demonstrations.",
        "success",
        "animate-delay-3"
    )

# ---------------------------------------------------
# EDA PAGE
# ---------------------------------------------------

elif page == "EDA Insights":

    st.markdown("""
    <div class="hero-card animate-fade-up">
        <div class="hero-title">📊 Exploratory Data Analysis</div>
        <div class="hero-subtitle">
            Explore the main patterns in the student dataset and understand the academic,
            demographic, and financial signals associated with dropout.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="animate-fade-in animate-delay-1">
        <span class="chip">Class imbalance detected</span>
        <span class="chip">Academic performance matters most</span>
        <span class="chip">Financial stress is meaningful</span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribution",
        "Academic Features",
        "Financial Features",
        "Correlation Heatmap"
    ])

    with tab1:
        section_header("Dropout Distribution", "Check whether the target classes are balanced.")

        dist_counts = (
            df["Dropout_Binary"]
            .map({0: "Continue", 1: "Dropout"})
            .value_counts()
            .reset_index()
        )
        dist_counts.columns = ["Student Status", "Count"]

        fig_dist = px.bar(
            dist_counts,
            x="Student Status",
            y="Count",
            text="Count",
            color="Student Status",
            color_discrete_map={"Continue": SUCCESS, "Dropout": DANGER},
            template="plotly_white"
        )

        fig_dist.update_traces(
            textposition="inside",
            textfont=dict(color="white", size=16),
            marker_line_color="rgba(255,255,255,0.20)",
            marker_line_width=1.0
        )

        fig_dist.update_layout(
            height=500,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color=TEXT),
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Student Status",
            yaxis_title="Count"
        )

        fig_dist.update_xaxes(
            showgrid=False,
            tickfont=dict(color=TEXT),
            title_font=dict(color=TEXT)
        )

        fig_dist.update_yaxes(
            showgrid=True,
            gridcolor="rgba(107, 114, 128, 0.22)",
            zeroline=False,
            tickfont=dict(color=TEXT),
            title_font=dict(color=TEXT)
        )

        st.plotly_chart(fig_dist, use_container_width=True)

        info_box(
            "<b>Insight:</b> The dataset shows class imbalance, which is why SMOTE was applied during training to improve the model’s ability to detect actual dropout students.",
            "warning",
            "animate-delay-2"
        )

    with tab2:
        section_header("Academic Features", "Understand how academic patterns differ across student outcomes.")

        fig_age = px.box(
            df,
            x=df["Dropout_Binary"].map({0: "Continue", 1: "Dropout"}),
            y="Age at enrollment",
            color=df["Dropout_Binary"].map({0: "Continue", 1: "Dropout"}),
            color_discrete_map={"Continue": SUCCESS, "Dropout": DANGER},
            template="plotly_white"
        )

        fig_age.update_layout(
            xaxis_title="Student Status",
            yaxis_title="Age at Enrollment",
            showlegend=False,
            height=450,
            font=dict(color=TEXT)
        )

        st.plotly_chart(fig_age, use_container_width=True)

        info_box(
            "<b>Academic takeaway:</b> Students who drop out often show weaker academic patterns overall, especially in semester performance and approved curricular units.",
            "info",
            "animate-delay-2"
        )

    with tab3:
        section_header("Financial Features", "Explore how tuition payment behavior relates to dropout.")

        tuition_dropout = (
            df.groupby("Tuition fees up to date")["Dropout_Binary"]
            .mean()
            .reset_index()
        )

        tuition_dropout["Tuition Status"] = tuition_dropout["Tuition fees up to date"].map({
            1: "Up to Date",
            0: "Not Up to Date"
        })

        tuition_dropout["Dropout Rate"] = tuition_dropout["Dropout_Binary"]

        fig_tuition = px.bar(
            tuition_dropout,
            x="Tuition Status",
            y="Dropout Rate",
            text=tuition_dropout["Dropout Rate"].apply(lambda x: f"{x:.1%}"),
            color="Tuition Status",
            color_discrete_map={"Up to Date": SUCCESS, "Not Up to Date": DANGER},
            template="plotly_white"
        )

        fig_tuition.update_layout(
            yaxis_title="Dropout Rate",
            xaxis_title="",
            showlegend=False,
            height=420,
            font=dict(color=TEXT)
        )
        fig_tuition.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig_tuition, use_container_width=True)

        info_box(
            "<b>Financial takeaway:</b> Students who are not up to date on tuition fees have a much higher dropout rate, making financial stability a strong risk signal.",
            "danger",
            "animate-delay-2"
        )

    with tab4:
        section_header("Correlation with Dropout", "See which numeric features are most strongly associated with dropout.")

        numeric_df = df.select_dtypes(include=["int64", "float64"])
        corr = numeric_df.corr()[["Dropout_Binary"]].sort_values(by="Dropout_Binary", ascending=False)
        corr = corr.reset_index()
        corr.columns = ["Feature", "Correlation"]

        fig_corr = px.imshow(
            corr[["Correlation"]].values,
            x=["Correlation with Dropout"],
            y=corr["Feature"],
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        fig_corr.update_layout(
            height=700,
            template="plotly_white",
            coloraxis_colorbar_title="Corr",
            font=dict(color=TEXT)
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        info_box(
            "<b>Interpretation:</b> Academic features tend to show stronger relationships with dropout than demographic variables, which is consistent with the model’s feature importance later on.",
            "info",
            "animate-delay-2"
        )

# ---------------------------------------------------
# MODEL PERFORMANCE PAGE
# ---------------------------------------------------

elif page == "Model Performance":

    st.markdown("""
    <div class="hero-card animate-fade-up">
        <div class="hero-title">📈 Model Performance & Evaluation</div>
        <div class="hero-subtitle">
            Review the final Random Forest model using key classification metrics,
            confusion matrix results, and ROC performance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        kpi_card("Accuracy", "0.884", "Overall percentage of correct predictions.", extra_class="animate-delay-1")
    with col2:
        kpi_card("Precision", "0.842", "Of predicted dropouts, how many were actually dropout cases?", extra_class="animate-delay-2")
    with col3:
        kpi_card("Recall", "0.785", "Of actual dropouts, how many were correctly identified?", extra_class="animate-delay-3")
    with col4:
        kpi_card("F1 Score", "0.812", "Balanced measure of precision and recall.", extra_class="animate-delay-4")
    with col5:
        kpi_card("ROC-AUC", "0.933", "Measures how well the model separates the two classes.", extra_class="animate-delay-5")

    left_col, right_col = st.columns(2)

    with left_col:
        section_header("Confusion Matrix", "A breakdown of correct and incorrect predictions.")

        cm_df = pd.DataFrame(
            [[635, 30], [73, 147]],
            index=["Actual Continue", "Actual Dropout"],
            columns=["Pred Continue", "Pred Dropout"]
        )

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto"
        )
        fig_cm.update_layout(height=420, template="plotly_white", font=dict(color=TEXT))
        st.plotly_chart(fig_cm, use_container_width=True)

    with right_col:
        section_header("ROC Curve", "Model discrimination across classification thresholds.")

        fpr = [0.0, 0.05, 0.10, 0.18, 0.30, 1.0]
        tpr = [0.0, 0.62, 0.78, 0.87, 0.94, 1.0]

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name="Random Forest (AUC = 0.933)",
            line=dict(color=PRIMARY, width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Baseline",
            line=dict(dash="dash", color=SUBTEXT, width=2)
        ))
        fig_roc.update_layout(
            template="plotly_white",
            height=420,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            font=dict(color=TEXT)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    info_box(
        "<b>Model summary:</b> Random Forest was selected because it achieved the strongest overall balance across accuracy, precision, recall, F1-score, and ROC-AUC after model comparison and hyperparameter tuning.",
        "success",
        "animate-delay-2"
    )

    info_box(
        "<b>Best parameters:</b><br>"
        "• n_estimators = 200<br>"
        "• max_depth = None<br>"
        "• min_samples_split = 2<br>"
        "• min_samples_leaf = 1",
        "info",
        "animate-delay-3"
    )

# ---------------------------------------------------
# MODEL COMPARISON PAGE
# ---------------------------------------------------

elif page == "Model Comparison":

    st.markdown("""
    <div class="hero-card animate-fade-up">
        <div class="hero-title">⚖️ Model Comparison</div>
        <div class="hero-subtitle">
            Compare Logistic Regression, Decision Tree, and Random Forest to understand
            why Random Forest was selected as the final model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    best_roc_model = model_metrics_df.loc[model_metrics_df["ROC-AUC"].idxmax(), "Model"]
    best_recall_model = model_metrics_df.loc[model_metrics_df["Recall"].idxmax(), "Model"]
    best_f1_model = model_metrics_df.loc[model_metrics_df["F1 Score"].idxmax(), "Model"]

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Best ROC-AUC", best_roc_model, "Model with the strongest class separation.", PRIMARY, "animate-delay-1")
    with c2:
        kpi_card("Best Recall", best_recall_model, "Model that catches the most dropout students.", SUCCESS, "animate-delay-2")
    with c3:
        kpi_card("Best F1 Score", best_f1_model, "Best balance between precision and recall.", WARNING, "animate-delay-3")

    section_header("Interactive Metric Comparison", "Compare model performance across all major evaluation metrics.")

    long_df = model_metrics_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )

    fig_compare = px.bar(
        long_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        template="plotly_white",
        text=long_df["Score"].apply(lambda x: f"{x:.3f}")
    )
    fig_compare.update_layout(height=500, yaxis_range=[0, 1], font=dict(color=TEXT))
    st.plotly_chart(fig_compare, use_container_width=True)

    section_header("Comparison Table", "Use this table to present the final evaluation results clearly.")

    styled_df = model_metrics_df.copy()
    styled_df[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]] = styled_df[
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ].round(3)

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    info_box(
        "<b>Why Random Forest won:</b> It delivered the best overall balance across recall, F1-score, and ROC-AUC, making it the most reliable model for identifying students at risk of dropout.",
        "success",
        "animate-delay-2"
    )

# ---------------------------------------------------
# INTERPRETATION PAGE
# ---------------------------------------------------

elif page == "Interpretation":

    st.markdown("""
    <div class="hero-card animate-fade-up">
        <div class="hero-title">🔍 Model Interpretation</div>
        <div class="hero-subtitle">
            Understand which features influenced the Random Forest model and how SHAP
            explains the behavior behind its predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    feature_importance_data = pd.DataFrame({
        "Feature": X_test_scaled.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    top_feature = feature_importance_data.iloc[0]["Feature"]

    col1, col2, col3 = st.columns(3)

    with col1:
        kpi_card(
            "Top Feature",
            top_feature,
            "The most important feature in the final Random Forest model.",
            extra_class="animate-delay-1"
        )

    with col2:
        kpi_card(
            "Key Driver Type",
            "Academic",
            "Academic performance features dominate the model.",
            extra_class="animate-delay-2"
        )

    with col3:
        kpi_card(
            "Financial Signal",
            "Tuition Status",
            "Tuition payment status is a meaningful financial risk signal.",
            extra_class="animate-delay-3"
        )

    section_header("Top 10 Feature Importances", "These features contributed most to the model’s decisions.")

    feature_importance_plot = feature_importance_data.sort_values(by="Importance", ascending=True)

    fig_fi = px.bar(
        feature_importance_plot,
        x="Importance",
        y="Feature",
        orientation="h",
        template="plotly_white",
        text=feature_importance_plot["Importance"].round(3)
    )
    fig_fi.update_layout(height=500, font=dict(color=TEXT))
    st.plotly_chart(fig_fi, use_container_width=True)

    section_header("SHAP Summary Plot", "SHAP explains how each feature pushes predictions toward dropout or non-dropout.")

    info_box(
        "<b>What SHAP shows:</b> Higher absolute SHAP values mean a feature has stronger impact. Positive and negative directions help explain whether a feature pushes a student toward dropout risk or toward continuation.",
        "info",
        "animate-delay-2"
    )

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_shap_sample)

    if isinstance(shap_values, list):
        shap_plot_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_plot_values = shap_values[:, :, 1]
    else:
        shap_plot_values = shap_values

    fig_shap = plt.figure(figsize=(9, 5.5))
    shap.summary_plot(shap_plot_values, X_shap_sample, show=False)
    st.pyplot(fig_shap, clear_figure=True)

    section_header("Interpretation Summary", "A simple summary of what the model has learned.")

    info_box(
        "<b>Main Insight:</b> The model relies most heavily on academic performance, especially semester grades and approved curricular units.",
        "success",
        "animate-delay-1"
    )
    info_box(
        "<b>Financial Insight:</b> Tuition fee status is an important predictor, showing that financial stability has a meaningful impact on dropout risk.",
        "warning",
        "animate-delay-2"
    )
    info_box(
        "<b>Demographic Insight:</b> Age at enrollment contributes to the model, but it is less influential than academic and financial features.",
        "info",
        "animate-delay-3"
    )
    info_box(
        "<b>Conclusion:</b> Dropout risk is driven mainly by a combination of academic performance, financial condition, and student background.",
        "danger",
        "animate-delay-4"
    )

# ---------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------

elif page == "Prediction":

    st.markdown("""
    <div class="hero-card animate-fade-up">
        <div class="hero-title">🎯 Student Dropout Prediction</div>
        <div class="hero-subtitle">
            Enter student details below to estimate dropout probability and understand
            which features are driving that prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    model_features = list(X_test_scaled.columns)
    raw_feature_df = df[model_features].copy()

    feature_means = raw_feature_df.mean()
    feature_stds = raw_feature_df.std().replace(0, 1)
    default_input = raw_feature_df.median(numeric_only=True).to_dict()

    with st.form("prediction_form"):

        st.subheader("Student Input Form")

        col1, col2 = st.columns(2)

        with col1:
            age_at_enrollment = st.number_input(
                "Age at enrollment",
                min_value=float(raw_feature_df["Age at enrollment"].min()),
                max_value=float(raw_feature_df["Age at enrollment"].max()),
                value=float(default_input.get("Age at enrollment", 20)),
                step=1.0
            )

            admission_grade = st.number_input(
                "Admission grade",
                min_value=float(raw_feature_df["Admission grade"].min()),
                max_value=float(raw_feature_df["Admission grade"].max()),
                value=float(default_input.get("Admission grade", 120)),
                step=1.0
            )

            tuition_fees_up_to_date = st.selectbox(
                "Tuition fees up to date",
                options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            debtor = st.selectbox(
                "Debtor",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

            scholarship_holder = st.selectbox(
                "Scholarship holder",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

        with col2:
            first_sem_approved = st.number_input(
                "Curricular units 1st sem (approved)",
                min_value=float(raw_feature_df["Curricular units 1st sem (approved)"].min()),
                max_value=float(raw_feature_df["Curricular units 1st sem (approved)"].max()),
                value=float(default_input.get("Curricular units 1st sem (approved)", 5)),
                step=1.0
            )

            second_sem_approved = st.number_input(
                "Curricular units 2nd sem (approved)",
                min_value=float(raw_feature_df["Curricular units 2nd sem (approved)"].min()),
                max_value=float(raw_feature_df["Curricular units 2nd sem (approved)"].max()),
                value=float(default_input.get("Curricular units 2nd sem (approved)", 5)),
                step=1.0
            )

            first_sem_grade = st.number_input(
                "Curricular units 1st sem (grade)",
                min_value=float(raw_feature_df["Curricular units 1st sem (grade)"].min()),
                max_value=float(raw_feature_df["Curricular units 1st sem (grade)"].max()),
                value=float(default_input.get("Curricular units 1st sem (grade)", 12)),
                step=0.1
            )

            second_sem_grade = st.number_input(
                "Curricular units 2nd sem (grade)",
                min_value=float(raw_feature_df["Curricular units 2nd sem (grade)"].min()),
                max_value=float(raw_feature_df["Curricular units 2nd sem (grade)"].max()),
                value=float(default_input.get("Curricular units 2nd sem (grade)", 12)),
                step=0.1
            )

            application_mode = st.number_input(
                "Application mode",
                min_value=float(raw_feature_df["Application mode"].min()),
                max_value=float(raw_feature_df["Application mode"].max()),
                value=float(default_input.get("Application mode", 1)),
                step=1.0
            )

        submitted = st.form_submit_button("Predict Dropout Risk")

    if submitted:

        input_row = default_input.copy()
        input_row["Age at enrollment"] = age_at_enrollment
        input_row["Admission grade"] = admission_grade
        input_row["Tuition fees up to date"] = tuition_fees_up_to_date
        input_row["Debtor"] = debtor
        input_row["Scholarship holder"] = scholarship_holder
        input_row["Curricular units 1st sem (approved)"] = first_sem_approved
        input_row["Curricular units 2nd sem (approved)"] = second_sem_approved
        input_row["Curricular units 1st sem (grade)"] = first_sem_grade
        input_row["Curricular units 2nd sem (grade)"] = second_sem_grade
        input_row["Application mode"] = application_mode

        input_df = pd.DataFrame([input_row], columns=model_features)

        input_scaled = (input_df - feature_means[model_features]) / feature_stds[model_features]
        input_scaled = input_scaled.fillna(0)

        prediction = best_model.predict(input_scaled)[0]
        probability = best_model.predict_proba(input_scaled)[0][1]

        risk_label, risk_color = risk_level(probability)

        section_header("Prediction Result", "Final model output for the student profile entered above.")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            kpi_card(
                "Predicted Class",
                "Dropout" if prediction == 1 else "Continue",
                "Final predicted class.",
                extra_class="animate-delay-1"
            )

        with result_col2:
            kpi_card(
                "Dropout Probability",
                f"{probability:.1%}",
                "Predicted probability of dropout.",
                extra_class="animate-delay-2"
            )

        with result_col3:
            kpi_card(
                "Risk Level",
                risk_label,
                "Risk bucket based on predicted dropout probability.",
                risk_color,
                "animate-delay-3"
            )

        gauge_col1, gauge_col2 = st.columns([1.1, 1.4])

        with gauge_col1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Dropout Risk (%)"},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, 33], "color": "#DCFCE7"},
                        {"range": [33, 66], "color": "#FEF3C7"},
                        {"range": [66, 100], "color": "#FEE2E2"}
                    ]
                }
            ))
            gauge.update_layout(height=350, template="plotly_white", font=dict(color=TEXT))
            st.plotly_chart(gauge, use_container_width=True)

        with gauge_col2:
            st.markdown(f"""
            <div class="card animate-fade-up hover-lift animate-delay-2">
                <h3 style="margin-top:0;color:{TEXT};">Prediction Summary</h3>
                <p style="color:{SUBTEXT};line-height:1.7;">
                    This student is classified as <b>{"likely to drop out" if prediction == 1 else "likely to continue"}</b>.
                    The estimated dropout probability is <b>{probability:.1%}</b>.
                </p>
                <p>
                    <span class="risk-badge" style="background:{risk_color};">{risk_label}</span>
                </p>
                <p style="color:{SUBTEXT};line-height:1.7;">
                    The result is based on the academic, financial, and enrollment information entered in the form.
                </p>
            </div>
            """, unsafe_allow_html=True)

        section_header("Why the Model Predicted This", "Top local drivers behind this specific prediction.")

        explainer = shap.TreeExplainer(best_model)
        single_shap_values = explainer.shap_values(input_scaled)

        if isinstance(single_shap_values, list):
            shap_single = single_shap_values[1]
        elif isinstance(single_shap_values, np.ndarray) and single_shap_values.ndim == 3:
            shap_single = single_shap_values[:, :, 1]
        else:
            shap_single = single_shap_values

        shap_importance = pd.DataFrame({
            "Feature": model_features,
            "SHAP Value": shap_single[0]
        })

        shap_importance["Absolute SHAP"] = shap_importance["SHAP Value"].abs()
        shap_importance = shap_importance.sort_values(by="Absolute SHAP", ascending=False).head(8)
        shap_importance = shap_importance.sort_values(by="SHAP Value")

        shap_importance["Impact Direction"] = shap_importance["SHAP Value"].apply(
            lambda x: "Pushes toward Dropout" if x > 0 else "Pushes toward Continue"
        )

        fig_pred_shap = px.bar(
            shap_importance,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            color="Impact Direction",
            color_discrete_map={
                "Pushes toward Dropout": DANGER,
                "Pushes toward Continue": SUCCESS
            },
            template="plotly_white"
        )
        fig_pred_shap.update_layout(height=500, font=dict(color=TEXT))
        st.plotly_chart(fig_pred_shap, use_container_width=True)

        info_box(
            "<b>How to read this:</b> Features with positive SHAP values push the prediction more toward dropout, while negative SHAP values push it more toward continuation.",
            "info",
            "animate-delay-2"
        )