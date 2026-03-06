import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title  = "Clinical Trial Dropout Predictor",
    page_icon   = "🏥",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

@st.cache_resource
def load_model():
    return joblib.load('models/xgboost_final_model.pkl')

@st.cache_data
def load_data():
    X_test     = pd.read_csv('data/X_test.csv')
    y_test     = pd.read_csv('data/y_test.csv').values.ravel()
    risk_scores = pd.read_csv('outputs/patient_risk_scores.csv')
    return X_test, y_test, risk_scores

model               = load_model()
X_test, y_test, risk_df = load_data()


st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/"
    "Agilisium_Logo.png/320px-Agilisium_Logo.png",
    use_column_width=True
)

st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Overview",
     "🔴 Patient Risk Table",
     "🧠 Model Insights",
     "🔮 Predict New Patient"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Clinical Trial Dropout Prediction")
st.sidebar.markdown("**Model:** XGBoost Classifier")
st.sidebar.markdown("**Intern:** Agilisium Remote Internship")


if page == "📊 Overview":
    st.title("📊 Clinical Trial Dropout — Overview")
    st.markdown("Real-time monitoring dashboard for patient retention in clinical trials.")
    st.markdown("---")

    total    = len(risk_df)
    high     = len(risk_df[risk_df['risk_level'] == '🔴 High'])
    medium   = len(risk_df[risk_df['risk_level'] == '🟡 Medium'])
    low      = len(risk_df[risk_df['risk_level'] == '🟢 Low'])
    avg_risk = risk_df['dropout_risk_%'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Patients",    total)
    col2.metric("🔴 High Risk",       high,   delta=f"{high/total*100:.1f}%")
    col3.metric("🟡 Medium Risk",     medium, delta=f"{medium/total*100:.1f}%")
    col4.metric("🟢 Low Risk",        low,    delta=f"{low/total*100:.1f}%")
    col5.metric("Avg Dropout Risk",  f"{avg_risk:.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Level Distribution")
        risk_counts = risk_df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        fig = px.pie(
            risk_counts,
            names  = 'Risk Level',
            values = 'Count',
            color  = 'Risk Level',
            color_discrete_map = {
                '🔴 High'  : '#e74c3c',
                '🟡 Medium': '#f39c12',
                '🟢 Low'   : '#2ecc71'
            },
            hole = 0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Dropout Risk Score Distribution")
        fig2 = px.histogram(
            risk_df,
            x      = 'dropout_risk_%',
            nbins  = 30,
            color_discrete_sequence = ['#3498db'],
            labels = {'dropout_risk_%': 'Dropout Risk (%)'}
        )
        fig2.add_vline(
            x          = avg_risk,
            line_dash  = "dash",
            line_color = "red",
            annotation_text = f"Avg: {avg_risk:.1f}%"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Actual vs Predicted Outcomes")
    outcome_counts = pd.Series(y_test).map(
        {0: 'Stayed', 1: 'Dropped Out'}
    ).value_counts().reset_index()
    outcome_counts.columns = ['Outcome', 'Count']
    fig3 = px.bar(
        outcome_counts,
        x     = 'Outcome',
        y     = 'Count',
        color = 'Outcome',
        color_discrete_map = {
            'Stayed'     : '#2ecc71',
            'Dropped Out': '#e74c3c'
        },
        text = 'Count'
    )
    fig3.update_traces(textposition='outside')
    st.plotly_chart(fig3, use_container_width=True)


elif page == "🔴 Patient Risk Table":
    st.title("🔴 Patient Risk Table")
    st.markdown("Filter and monitor individual patient dropout risk scores.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options  = ['🔴 High', '🟡 Medium', '🟢 Low'],
            default  = ['🔴 High', '🟡 Medium', '🟢 Low']
        )
    with col2:
        min_risk, max_risk = st.slider(
            "Dropout Risk % Range",
            min_value = 0,
            max_value = 100,
            value     = (0, 100)
        )

    filtered = risk_df[
        (risk_df['risk_level'].isin(risk_filter)) &
        (risk_df['dropout_risk_%'] >= min_risk) &
        (risk_df['dropout_risk_%'] <= max_risk)
    ]

    st.markdown(f"**Showing {len(filtered)} patients**")

    def color_risk(val):
        if '🔴' in str(val):
            return 'background-color: #fadbd8'
        elif '🟡' in str(val):
            return 'background-color: #fef9e7'
        elif '🟢' in str(val):
            return 'background-color: #eafaf1'
        return ''

    styled = filtered.style.applymap(
        color_risk, subset=['risk_level']
    )
    st.dataframe(styled, use_container_width=True, height=500)

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label     = "⬇️ Download Filtered Table as CSV",
        data      = csv,
        file_name = "filtered_patient_risks.csv",
        mime      = "text/csv"
    )


elif page == "🧠 Model Insights":
    st.title("🧠 Model Insights")
    st.markdown("Understand what drives dropout predictions.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📊 Feature Importance", "🔍 SHAP Analysis"])

    with tab1:
        st.subheader("Top Features Driving Dropout")
        importance_df = pd.DataFrame({
            'Feature'   : X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig = px.bar(
            importance_df,
            x     = 'Importance',
            y     = 'Feature',
            orientation = 'h',
            color = 'Importance',
            color_continuous_scale = 'Reds',
            title = 'Feature Importance Scores'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📖 What Does Each Feature Mean?")
        feature_explanations = {
            "visit_compliance_rate" : "% of scheduled visits the patient completed — lower = higher dropout risk",
            "visits_missed"         : "Number of visits missed — key dropout signal",
            "trial_burden_score"    : "Combined score of adverse events + protocol deviations",
            "site_distance_km"      : "Distance from patient's home to trial site",
            "adverse_events"        : "Number of side effects reported",
            "days_in_trial"         : "How long the patient has been in the trial",
            "age"                   : "Patient age — older patients may have more constraints",
            "high_risk_flag"        : "1 if patient is both far from site AND missing many visits",
        }
        for feat, explanation in feature_explanations.items():
            if feat in importance_df['Feature'].values:
                st.markdown(f"**{feat}** → {explanation}")

    with tab2:
        st.subheader("SHAP Summary — Direction of Feature Impact")
        st.markdown(
            "🔴 Red = high feature value | 🔵 Blue = low feature value | "
            "Right = increases dropout risk | Left = decreases dropout risk"
        )

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        fig_shap, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()


elif page == "🔮 Predict New Patient":
    st.title("🔮 Predict Dropout Risk for a New Patient")
    st.markdown("Enter patient details below to get their dropout risk score.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Patient Info")
        age               = st.slider("Age", 18, 75, 45)
        gender            = st.selectbox("Gender", ["Male", "Female"])
        bmi               = st.slider("BMI", 17.0, 40.0, 25.0)
        employment_status = st.selectbox(
            "Employment Status",
            ["Employed", "Unemployed", "Retired"]
        )
        has_caregiver     = st.selectbox("Has Caregiver?", [0, 1])
        insurance_coverage= st.selectbox("Has Insurance?", [0, 1])

    with col2:
        st.subheader("🏥 Trial Info")
        trial_phase    = st.selectbox(
            "Trial Phase", ["Phase I", "Phase II", "Phase III"]
        )
        disease_type   = st.selectbox(
            "Disease Type",
            ["Oncology", "Cardiology", "Neurology", "Diabetes"]
        )
        treatment_arm  = st.selectbox("Treatment Arm", ["Drug", "Placebo"])
        site_distance  = st.slider("Site Distance (km)", 1, 150, 30)
        days_in_trial  = st.slider("Days in Trial", 10, 180, 60)

    with col3:
        st.subheader("📋 Engagement")
        visits_completed     = st.slider("Visits Completed", 1, 12, 6)
        visits_missed        = st.slider("Visits Missed", 0, 6, 1)
        adverse_events       = st.slider("Adverse Events", 0, 5, 1)
        protocol_deviations  = st.slider("Protocol Deviations", 0, 3, 0)

    st.markdown("---")

    if st.button("🔮 Predict Dropout Risk", use_container_width=True):

        gender_enc     = 1 if gender == "Male" else 0
        employment_enc = {"Employed": 0, "Unemployed": 2, "Retired": 1}[employment_status]
        trial_enc      = {"Phase I": 0, "Phase II": 1, "Phase III": 2}[trial_phase]
        disease_enc    = {"Cardiology": 0, "Diabetes": 1, "Neurology": 2, "Oncology": 3}[disease_type]
        treatment_enc  = 0 if treatment_arm == "Drug" else 1

        visit_compliance  = visits_completed / (visits_completed + visits_missed + 1)
        trial_burden      = adverse_events + (protocol_deviations * 2)
        high_risk_flag    = 1 if (site_distance > 75 and visits_missed > 3) else 0

        input_data = pd.DataFrame([{
            'age'                  : age,
            'gender'               : gender_enc,
            'bmi'                  : bmi,
            'trial_phase'          : trial_enc,
            'disease_type'         : disease_enc,
            'treatment_arm'        : treatment_enc,
            'site_distance_km'     : site_distance,
            'visits_completed'     : visits_completed,
            'visits_missed'        : visits_missed,
            'adverse_events'       : adverse_events,
            'protocol_deviations'  : protocol_deviations,
            'days_in_trial'        : days_in_trial,
            'employment_status'    : employment_enc,
            'has_caregiver'        : has_caregiver,
            'insurance_coverage'   : insurance_coverage,
            'visit_compliance_rate': visit_compliance,
            'trial_burden_score'   : trial_burden,
            'high_risk_flag'       : high_risk_flag,
        }])

        input_data = input_data[X_test.columns]

        risk_prob = model.predict_proba(input_data)[0][1]
        risk_pct  = risk_prob * 100

        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res2:
            if risk_pct >= 66:
                st.error(f"🔴 HIGH RISK\n\n**Dropout Probability: {risk_pct:.1f}%**")
                st.markdown("⚠️ Immediate intervention recommended. Contact patient within 48 hours.")
            elif risk_pct >= 33:
                st.warning(f"🟡 MEDIUM RISK\n\n**Dropout Probability: {risk_pct:.1f}%**")
                st.markdown("📞 Schedule a check-in call with this patient soon.")
            else:
                st.success(f"🟢 LOW RISK\n\n**Dropout Probability: {risk_pct:.1f}%**")
                st.markdown("✅ Patient appears engaged. Continue standard monitoring.")

        st.markdown("---")
        st.subheader("🔍 Why This Prediction? (SHAP Explanation)")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(
            shap.Explanation(
                values        = shap_values[0],
                base_values   = explainer.expected_value,
                data          = input_data.iloc[0],
                feature_names = X_test.columns.tolist()
            ),
            show = False
        )
        plt.tight_layout()
        st.pyplot(fig)
