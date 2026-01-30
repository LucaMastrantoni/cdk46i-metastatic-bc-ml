import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import utils

import warnings
warnings.filterwarnings('ignore')

# Load the model
cox_model = joblib.load('Models/coxnet_model.joblib')
gbm_model = joblib.load('Models/gbm_model.joblib')

# Title
st.header("""Predicting Progression-Free Survival in Hormone-Receptor Positive Metastatic Breast Cancer treated with CDK4/6 Inhibitors""")
st.markdown("""Graphical interface for "Predicting Progression-Free Survival in Hormone-Receptor Positive Metastatic Breast Cancer treated with CDK4/6 Inhibitors: A Machine Learning Approach". The Breast, 2026. https://doi.org/10.1016/j.breast.2026.104715""")

# User inputs
st.divider()
st.subheader('Patient characteristics')

Age = st.number_input("Age", min_value=35, max_value=85, step=1, value=60 )
Menopausal = st.radio("Menopausal Status", ("Post Menopausal", "Pre Menopausal"))
Hormone_Resistance = st.radio("Hormone Resistance", ("Primary Resistant", "Secondary Resistant", "Hormone-Sensitive"))
ER = st.number_input("ER (%)", 0, 100, step=1, value=50)
PgR = st.number_input("PgR (%)", 0, 100, step=1, value=30)
HER2 = st.radio("HER2 Status", ("Low", "Zero"))
Ki67 = st.number_input("Ki67 (%)", 0, 100, step=1, value=40)
Histotype = st.radio("Histotype", ("Ductal", "Non Ductal"))
M_Brain = st.selectbox("Brain Metastases", ("No", "Yes"))
M_Liver = st.selectbox("Liver Metastases", ("No", "Yes"))
M_Bone = st.selectbox("Bone Metastases", ("No", "Yes"))
M_Peritoneal = st.selectbox("Peritoneal Metastases", ("No", "Yes"))
Bone_Only = st.selectbox("Bone Only Disease", ("No", "Yes"))
Sinc = st.selectbox("Synchronous", ("No", "Yes"))

st.space()
with st.expander("⚙️ Advanced", expanded=False):
    model_choos = st.selectbox("Model", ("GBM"), help="Model choice for survival prediction. Only GBM (Gradient Boosting Machines) is available for risk stratification in app at the moment")
    cutoff_risk = st.selectbox("Cut-off", ("GMM", "MSRS"), help="Cut-off choice for risk group stratification (GMM: Gaussian Mixture Model or MSRS: Maximally Selected Rank Statistics). Default is GMM.")
    last_time = st.number_input("Maxtime (months)", min_value=6, max_value=90, value=60, step=3, help="Optional: Right cut of the months axis for survival plot")
    target_time = st.number_input("Target time (months)", min_value=6, max_value=60, value=6, step=3, help="Optional: a specific time point to evaluate survival probability")

# Create a DataFrame with user inputs (binary variables converted to 0/1)
input_data = utils.create_input_data(Age, Menopausal, Hormone_Resistance, ER, PgR, HER2, Ki67, Histotype, M_Brain, M_Liver, M_Bone, M_Peritoneal, Bone_Only, Sinc)

# select cox or gbm according to user choice
model = utils.model_choice(model_choos, cox_model, gbm_model)
cutoff = utils.surv_cutoff(cutoff_risk)
cut_off_F1 = 0.2032
time_points = np.linspace(0, last_time, 100)

st.space()
st.subheader('Predict')
# Prediction
if st.button("Predict survival and compute SHAP explanation"):
    try:
        st.divider()
        surv_fns = model.predict_survival_function(input_data)
        risk_score = model.predict(input_data)

        median_time = utils.survival_time50(surv_fns, time_points)
        surv_at_target_months = np.array([fn(target_time) for fn in surv_fns])
        risk_group = utils.risk_group(risk_score, cutoff)

        surv_at_6_months = np.array([fn(6) for fn in surv_fns])
        risk_6_months = 1 - surv_at_6_months
        early_prog = utils.early_progression(risk_6_months, cut_off_F1)

        st.subheader("Report")

        st.write(f"{Age} years old, {Menopausal.lower()} woman with {Hormone_Resistance.lower()} breast cancer, {Histotype.lower()} histotype. ER: {ER}%, PgR: {PgR}%, HER2 {HER2}, Ki67: {Ki67}%.") 
        st.write(f"Metastases in brain: {M_Brain.lower()}, liver: {M_Liver.lower()}, bone: {M_Bone.lower()}, peritoneum: {M_Peritoneal.lower()}. Bone only disease: {Bone_Only.lower()}. Synchronous: {Sinc.lower()}")
        
        st.subheader(f"Risk group: {risk_group}")
        st.write(f"The 50% PFS probability is at {median_time:.1f} months")
        st.write(f"The {target_time} months survival probability is {surv_at_target_months.flat[0]*100:.1f}%")

        st.subheader("Survival probability")
        fig = utils.plot_survival_functions(surv_fns, model_choos, time_points)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader(f"Early progression at 6 months: {early_prog}")
        st.write(f"The 6 months early progression probability is {risk_6_months.flat[0]*100:.1f}%")

        st.subheader(f"SHAP Explanation")
        sample, input_data_ord, model = utils.preprocess_shap(input_data, gbm_model)
        exp = utils.shap_explanation(sample, input_data_ord, model)
        fig_shap = utils.plot_shap(exp)
        st.pyplot(fig_shap)
        plt.close(fig_shap)

    except Exception as e:
        st.write(f"An error occurred during prediction: {e}")

#### END
st.space()
st.divider()
st.subheader(f"Acknowledgements")
st.markdown("""This study was conducted at Comprehensive Cancer Center, Fondazione Policlinico Universitario Agostino Gemelli IRCCS (Rome, Italy) and Ospedale Isola Tiberina - Gemelli Isola, (Rome, Italy).""")         
st.image("App/logo.PNG")
st.markdown("""Web add developed by Luca Mastrantoni. Source code available at https://github.com/LucaMastrantoni/cdk46i-metastatic-bc-ml.git""")




















