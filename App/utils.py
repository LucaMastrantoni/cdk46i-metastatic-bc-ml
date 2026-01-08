import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

###
feature_names = ['Age', 'ER', 'PgR', 'Ki67', 'Menopausal', 
                 'Hormone_Resistance_Primary', 'Hormone_Resistance_Secondary', 'HER2', 'Histotype',
                 'M_Brain', 'M_Liver', 'M_Bone', 'M_Peritoneal', 'Bone_Only', 'Sinc']
nicer_feature_names = ['Age', 'ER', 'PgR', 'Ki67', 'Menopausal Status', 
                       'Primary Hormone Resistance', 'Secondary Hormone Resistance', 'HER2 Status', 'Histotype',
                       'Brain Metastases', 'Liver Metastases', 'Bone Metastases', 'Peritoneal Metastases', 'Bone Only Disease', 'Synchronous']

# Data frame creation
def create_input_data(Age, Menopausal, Hormone_Resistance, ER, PgR, HER2, Ki67, Histotype, M_Brain, M_Liver, M_Bone, M_Peritoneal, Bone_Only, Sinc):
    input_data = pd.DataFrame({
        'Age': [Age],  # Fixed age as in the training set
        'Menopausal': [1 if Menopausal == "Post Menopausal" else 0],
        'Hormone_Resistance_Primary': [1 if Hormone_Resistance == "Primary" else 0],
        'Hormone_Resistance_Secondary': [1 if Hormone_Resistance == "Secondary" else 0],
        'ER': [ER], 
        'PgR': [PgR],
        'HER2': [1 if HER2 == "Zero" else 0],
        'Ki67': [Ki67],
        'Histotype': [1 if Histotype == "Non Ductal" else 0],
        'M_Brain': [1 if M_Brain == "Yes" else 0],
        'M_Liver': [1 if M_Liver == "Yes" else 0],
        'M_Bone': [1 if M_Bone == "Yes" else 0],
        'M_Peritoneal': [1 if M_Peritoneal == "Yes" else 0],
        'Bone_Only': [1 if Bone_Only == "Yes" else 0],
        'Sinc': [1 if Sinc == "Yes" else 0]   
})
    return input_data

# Helpers
def model_choice(model_choos, cox_model, gbm_model):
    if model_choos == "CoxNet":
        model = cox_model
    else:
        model = gbm_model
    return model

def surv_cutoff(cutoff_risk):
    if cutoff_risk == "GMM":
        cutoff = 0.1432
    else:
        cutoff = 0.2030
    return cutoff

def risk_group(risk_score, cutoff):
    if risk_score > cutoff:
        risk_group = "High Risk"
    else:
        risk_group = "Low Risk"
    return risk_group

def early_progression(risk_6_months, cut_off_F1):
    if risk_6_months > cut_off_F1:
        early_prog = "High Risk"
    else:
        early_prog = "Low Risk"
        return early_prog
    
# Function to plot survival functions
def plot_survival_functions(surv_fns, model_choos, time_points):
    fig, ax = plt.subplots(figsize=(10, 6))
    fn = surv_fns[0]  # single patient
    ax.step( time_points,
        fn(time_points),
        where="post",
        linewidth=2,
        alpha=0.9,
        label=model_choos,
        color= "#c64436" if model_choos == "CoxNet" else "#664797")

    ax.set_title("Predicted Survival Function")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Predicted PFS probability")
    ax.set_xticks(range(0, 66, 6))
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.legend()

    plt.tight_layout()
    return fig

def survival_time50(surv_fns, time_points):
    surv = surv_fns[0](time_points)
    median_time = time_points[np.argmin(np.abs(surv - 0.5))]
    return median_time


# Functions for SHAP explanation
def preprocess_shap(input_data, gbm_model):
    preprocess = gbm_model.named_steps["columntransformer"]
    scaler = gbm_model.named_steps["standardscaler"]
    model = gbm_model.named_steps["gradientboostingsurvivalanalysis"]

    sample = preprocess.transform(input_data)
    sample = scaler.transform(sample)

    sample_data = pd.DataFrame(sample, columns=feature_names)
    input_data_ord = pd.DataFrame(input_data,columns=feature_names)

    return sample_data, input_data_ord, model

def shap_explanation(sample_data, input_data_ord, model):
    background = joblib.load("shap_background_gbm.pkl")

    explainer = shap.KernelExplainer(lambda X: model.predict(X), background, feature_names=nicer_feature_names)
    shap_values = explainer.shap_values(sample_data, nsamples=200)
    exp = shap.Explanation(
        values=shap_values,               # SHAP for linear predictor
        base_values=explainer.expected_value,
        data=input_data_ord,                     # NumPy array OK
        feature_names=nicer_feature_names)
    return exp

def plot_shap(exp):
    shap.plots.waterfall(exp[0], max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 4)   # width, height in inches
    return fig