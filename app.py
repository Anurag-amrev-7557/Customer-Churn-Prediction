"""Interactive Customer Churn Analysis & Retention Strategy Dashboard."""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.business_impact import calculate_decile_table, simulate_retention_campaign

st.set_page_config(
    page_title="Telco Churn Analytics & Retention Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive look & feel
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .callout-box {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 12px 0;
    }
    .callout-warning {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data_and_reports():
    df = pd.read_csv("data/processed/telco_churn_cleaned.csv")
    df_preds = pd.read_csv("models/test_predictions.csv")

    with open("reports/eda_summary.json") as f:
        eda_summary = json.load(f)

    with open("models/model_comparison.json") as f:
        model_comp = json.load(f)

    with open("reports/explainability_summary.json") as f:
        explain_summary = json.load(f)

    with open("reports/retention_strategy_summary.json") as f:
        retention_summary = json.load(f)

    return df, df_preds, eda_summary, model_comp, explain_summary, retention_summary


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")


try:
    df, df_preds, eda_summary, model_comp, explain_summary, retention_summary = load_data_and_reports()
    model = load_model()
except Exception as e:
    st.error(f"Error loading project artifacts: {e}. Please ensure pipeline has executed.")
    st.stop()

# Header
st.markdown('<div class="main-header">Customer Churn Analysis & Retention Strategy</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">IBM Telco Dataset (7,043 Customers) • End-to-End Machine Learning, Segmentation & ROI Prioritization</div>',
    unsafe_allow_html=True,
)

# Top KPI metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Total Customer Records</div>
        <div class="metric-value">7,043</div>
        <div style="color: #64748b; font-size: 0.8rem;">100% Validated (0 duplicates)</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Baseline Churn Rate</div>
        <div class="metric-value">{df['Churn_Binary'].mean()*100:.1f}%</div>
        <div style="color: #ef4444; font-size: 0.8rem;">1,869 churned customers</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Best Model ROC-AUC</div>
        <div class="metric-value">{model_comp['best_roc_auc']:.3f}</div>
        <div style="color: #10b981; font-size: 0.8rem;">{model_comp['best_model_name']}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    roi_val = retention_summary["campaign_simulation_top_20pct"]["roi_pct"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Top 20% Retention ROI</div>
        <div class="metric-value">+{roi_val:.0f}%</div>
        <div style="color: #10b981; font-size: 0.8rem;">Captures 50.3% of churners</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Tabs
tabs = st.tabs([
    "🎯 Executive Insights & Business Answers",
    "📊 Customer Segmentation & EDA",
    "🤖 Model Benchmarking & Drivers",
    "💰 Retention ROI Simulator & Live Scoring",
])

# ----------------- TAB 1: EXECUTIVE INSIGHTS -----------------
with tabs[0]:
    st.subheader("Direct Answers to Core Business Questions")

    st.markdown("""
    #### 1. Which customer segments are most likely to churn?
    * **Month-to-Month Contract Customers**: Have a **42.7% churn rate**, compared to **11.3%** for 1-year contracts and just **2.8%** for 2-year contracts.
    * **Fiber Optic Internet Users**: Experience **41.9% churn** (vs. 18.9% for DSL and 7.4% for no internet), driven by high monthly rates and competitive fiber alternatives.
    * **Electronic Check Payers**: Suffer **45.3% churn**, contrasting sharply with bank transfers (16.7%) and credit cards (15.2%).
    * **New Customers (0-12 Months Tenure)**: Account for a **47.4% churn rate**, dropping down to **6.6%** once customers exceed 48 months.
    * **Senior Citizens**: Experience **41.7% churn** vs 23.6% for non-seniors.
    """)

    st.markdown("""
    #### 2. Which customer characteristics are most strongly associated with churn?
    * **Tenure (r = -0.352, Odds Ratio = 0.285)**: The single strongest protective attribute. Each unit increase in tenure slashes churn odds by **71.5%**.
    * **Contract Structure (Cramér's V = 0.410)**: Month-to-month contracts elevate churn risk by **+79.7%** (Odds Ratio = 1.797).
    * **Internet Service Type (Cramér's V = 0.323)**: Fiber optic subscription nearly doubles churn odds (**Odds Ratio = 1.912**).
    * **Lack of TechSupport & OnlineSecurity (Cramér's V ~ 0.345)**: Customers without bundled protection services churn at 3-4x higher rates.
    """)

    st.markdown("""
    #### 3. How can a telecom company use churn predictions to prioritize customer-retention efforts?
    * **Concentrated Risk Targeting**: By targeting only the **top 20% highest-risk customers** identified by the model, the company captures **50.5% of all churners** (2.5x more efficient than random contact).
    * **Substantial Financial ROI**: In our 1,409-customer test set alone, targeting the top 20% with a $40 retention offer and 30% save rate produces **$41,011 in net profit** with an **ROI of 364.9%** (a **+$35,042 lift** over non-targeted outreach).
    * **Tiered Intervention Playbooks**: Avoid discounting low-risk accounts; reserve rich contract extension incentives for high-value month-to-month fiber customers.
    """)

    st.divider()
    st.subheader("Strategic Retention Playbook")

    playbook_df = pd.DataFrame(retention_summary["retention_playbooks"])
    st.table(playbook_df.rename(columns={
        "tier": "Risk Tier",
        "profile": "Customer Segment Profile",
        "action": "Recommended Retention Strategy",
        "target_save_rate": "Expected Save Rate",
    }))

# ----------------- TAB 2: EDA & SEGMENTATION -----------------
with tabs[1]:
    st.subheader("Interactive Customer Segmentation & Statistical Associations")

    col_left, col_right = st.columns(2)

    with col_left:
        seg_choice = st.selectbox(
            "Select Segment to Inspect:",
            ["Contract", "InternetService", "PaymentMethod", "TenureCohort", "SeniorCitizen", "TechSupport"],
            index=0,
        )

        seg_data = pd.DataFrame(eda_summary["segment_profiles"][seg_choice])
        fig_seg = px.bar(
            seg_data,
            x=seg_choice,
            y="churn_rate",
            color="churn_rate",
            color_continuous_scale="Reds",
            text="churn_rate",
            title=f"Churn Rate by {seg_choice}",
            labels={"churn_rate": "Churn Rate"},
        )
        fig_seg.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_seg.update_layout(yaxis=dict(tickformat=".0%", range=[0, max(seg_data['churn_rate']) * 1.25]))
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_right:
        # Cramér's V Leaderboard
        assoc = eda_summary["associations"]["categorical_cramers_v"]
        df_assoc = pd.DataFrame(list(assoc.items()), columns=["Characteristic", "Cramér's V"]).sort_values(by="Cramér's V")

        fig_assoc = px.bar(
            df_assoc,
            x="Cramér's V",
            y="Characteristic",
            orientation="h",
            color="Cramér's V",
            color_continuous_scale="Viridis",
            title="Categorical Association with Churn (Cramér's V)",
        )
        st.plotly_chart(fig_assoc, use_container_width=True)

    # Monthly Charges and Tenure vs Churn
    st.subheader("Tenure vs. Monthly Charges by Churn Status")
    sample_sub = df.sample(n=min(1500, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample_sub,
        x="tenure",
        y="MonthlyCharges",
        color="Churn",
        color_discrete_map={"No": "#3b82f6", "Yes": "#ef4444"},
        opacity=0.6,
        title="Customer Landscape: Monthly Charges vs. Tenure (Sampled 1,500 records)",
        labels={"tenure": "Tenure (Months)", "MonthlyCharges": "Monthly Charges ($)"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------- TAB 3: MODEL PERFORMANCE & EXPLAINABILITY -----------------
with tabs[2]:
    st.subheader("Model Benchmarking & Evaluation")

    # Benchmarking Table
    bench_records = []
    for m_name, m_res in model_comp["results"].items():
        bench_records.append({
            "Model Pipeline": m_name,
            "CV ROC-AUC": m_res["cross_validation"]["cv_roc_auc"],
            "Test ROC-AUC": m_res["test_metrics"]["roc_auc"],
            "Precision": m_res["test_metrics"]["precision"],
            "Recall": m_res["test_metrics"]["recall"],
            "F1-Score": m_res["test_metrics"]["f1_score"],
            "PR-AUC": m_res["test_metrics"]["pr_auc"],
            "Brier Score": m_res["test_metrics"]["brier_score"],
        })
    df_bench = pd.DataFrame(bench_records).sort_values(by="Test ROC-AUC", ascending=False)
    st.dataframe(df_bench.style.highlight_max(subset=["Test ROC-AUC", "Recall", "F1-Score"], color="#dcfce7"), use_container_width=True)

    c_roc, c_pr = st.columns(2)

    with c_roc:
        # ROC Curves
        fig_roc = go.Figure()
        for m_name, c_data in model_comp["curves_data"].items():
            fig_roc.add_trace(go.Scatter(x=c_data["fpr"], y=c_data["tpr"], mode="lines", name=m_name))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
        fig_roc.update_layout(
            title="ROC Curves on Held-Out Test Set",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with c_pr:
        # Precision-Recall Curves
        fig_pr = go.Figure()
        for m_name, c_data in model_comp["curves_data"].items():
            fig_pr.add_trace(go.Scatter(x=c_data["recall"], y=c_data["precision"], mode="lines", name=m_name))
        fig_pr.update_layout(
            title="Precision-Recall Curves on Held-Out Test Set",
            xaxis_title="Recall",
            yaxis_title="Precision",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.subheader("Top Feature Importance & Odds Ratios")
    c_drivers, c_protect = st.columns(2)

    with c_drivers:
        df_risk = pd.DataFrame(explain_summary["top_churn_risk_drivers"]).head(7)
        fig_risk = px.bar(
            df_risk,
            x="odds_ratio",
            y="feature",
            orientation="h",
            color="odds_ratio",
            color_continuous_scale="Oranges",
            title="Top Risk Drivers (Odds Ratio > 1)",
            labels={"odds_ratio": "Odds Ratio", "feature": "Feature"},
        )
        fig_risk.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_risk, use_container_width=True)

    with c_protect:
        df_prot = pd.DataFrame(explain_summary["top_churn_protective_factors"]).head(7)
        fig_prot = px.bar(
            df_prot,
            x="odds_ratio",
            y="feature",
            orientation="h",
            color="odds_ratio",
            color_continuous_scale="Blues_r",
            title="Top Protective Factors (Odds Ratio < 1)",
            labels={"odds_ratio": "Odds Ratio", "feature": "Feature"},
        )
        fig_prot.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_prot, use_container_width=True)

# ----------------- TAB 4: RETENTION SIMULATOR & SCORING -----------------
with tabs[3]:
    st.subheader("Customer Retention Campaign Financial Simulator")
    st.markdown("Model the economic trade-offs of targeting high-probability churn candidates based on your budget.")

    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        target_pct = st.slider("Target Top % Highest-Risk Customers", min_value=5, max_value=50, value=20, step=5) / 100.0
    with s_col2:
        offer_cost = st.slider("Retention Offer Cost per Customer ($)", min_value=10, max_value=150, value=40, step=5)
    with s_col3:
        save_rate = st.slider("Customer Save Rate (% of churners retained)", min_value=10, max_value=60, value=30, step=5) / 100.0

    sim_res = simulate_retention_campaign(df_preds, target_top_pct=target_pct, offer_cost=offer_cost, save_rate=save_rate)

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Customers Contacted", f"{sim_res['customers_targeted']:,} / {sim_res['total_test_customers']}")
    rc2.metric("Churners Captured", f"{sim_res['actual_churners_captured']:,} ({sim_res['churners_captured_pct']:.1f}%)")
    rc3.metric("Net Financial Gain", f"${sim_res['net_profit']:,.2f}")
    rc4.metric("Campaign ROI", f"{sim_res['roi_pct']:.1f}%")

    st.markdown(f"""
    <div class="callout-box">
        <b>Financial Lift:</b> By using machine learning model prioritization instead of random customer outreach, 
        the business achieves an additional <b>+${sim_res['lift_over_random']:,.2f}</b> in net profit!
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Individual Customer Live Scoring
    st.subheader("⚡ Live Individual Customer Churn Scorer")
    st.markdown("Simulate a customer profile to predict churn probability and receive instant retention guidance.")

    form_col1, form_col2, form_col3 = st.columns(3)

    with form_col1:
        in_tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=4)
        in_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        in_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        in_payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    with form_col2:
        in_monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=130.0, value=85.0)
        in_tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        in_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        in_paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    with form_col3:
        in_senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        in_partner = st.selectbox("Partner", ["No", "Yes"])
        in_dependents = st.selectbox("Dependents", ["No", "Yes"])
        in_phone = st.selectbox("Phone Service", ["Yes", "No"])

    # Compute TotalCharges approximation
    in_total = in_monthly * max(1, in_tenure)

    input_data = pd.DataFrame([{
        "tenure": in_tenure,
        "MonthlyCharges": in_monthly,
        "TotalCharges": in_total,
        "gender": "Female",
        "SeniorCitizen": in_senior,
        "Partner": in_partner,
        "Dependents": in_dependents,
        "PhoneService": in_phone,
        "MultipleLines": "No",
        "InternetService": in_internet,
        "OnlineSecurity": in_security,
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": in_tech,
        "StreamingTV": "Yes" if in_internet != "No" else "No internet service",
        "StreamingMovies": "Yes" if in_internet != "No" else "No internet service",
        "Contract": in_contract,
        "PaperlessBilling": in_paperless,
        "PaymentMethod": in_payment,
    }])

    prob = float(model.predict_proba(input_data)[0, 1])

    g_col1, g_col2 = st.columns([1, 2])
    with g_col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ef4444" if prob > 0.5 else "#3b82f6"},
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            title={"text": "Predicted Churn Probability"},
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with g_col2:
        st.markdown("#### Recommended Retention Playbook")
        if prob >= 0.60:
            st.error("🚨 **CRITICAL FLIGHT RISK**: Customer has >60% probability of churning.")
            st.markdown("""
            * **Immediate Action**: Dispatch to Senior Retention Specialist.
            * **Offer**: Provide a 1-year contract lock incentive ($15/month discount for 12 months).
            * **Add-on**: Include 6 months of complimentary Tech Support & Online Security to improve service stickiness.
            """)
        elif prob >= 0.35:
            st.warning("⚠️ **MODERATE RISK**: Customer shows early attrition signals.")
            st.markdown("""
            * **Action**: Automated engagement sequence.
            * **Offer**: $5/month discount for enrolling in Autopay via Credit Card/Bank Transfer.
            * **Check-in**: Send satisfaction survey regarding internet stability and speed.
            """)
        else:
            st.success("✅ **HEALTHY CUSTOMER**: Low churn likelihood.")
            st.markdown("""
            * **Action**: Do not offer discounts (prevents margin erosion).
            * **Nurture**: Deliver loyalty recognition and explore relevant service upgrades (e.g. streaming add-ons).
            """)
