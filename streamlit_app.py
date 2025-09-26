# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shap

st.set_page_config(page_title="Insurance Charges Dashboard", layout="wide", page_icon="💰")

# -----------------------------
# Load trained artifacts
# -----------------------------
xg = joblib.load("final_insurance_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "KPIs", "Prediction", "Risk Analysis", "Download"])

# -----------------------------
# Load Data
# -----------------------------
DATA_PATH = "insurance.csv"
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# One-hot encode categorical features for consistency
cat_cols = ['sex', 'smoker', 'region']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Add smoker_bmi and bmi_group features for prediction
df_encoded['smoker_bmi'] = df_encoded.get('smoker_yes', 0) * df['bmi']
df_encoded['bmi_group'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 40, 100],
                                 labels=['underweight','normal','overweight','obese','extreme'])
df_encoded = pd.get_dummies(df_encoded, columns=['bmi_group'], drop_first=True)

# -----------------------------
# EDA Page
# -----------------------------
if page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("### Data Overview")
    st.write(df.head())
    st.write(df.describe())

    st.markdown("### Numeric Feature Distributions")
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    fig, axes = plt.subplots(1, len(num_cols), figsize=(15,4))
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], ax=axes[i], kde=True, color='teal')
        axes[i].set_title(col)
    st.pyplot(fig)

    st.markdown("### Categorical Feature Counts")
    for col in ['sex', 'smoker', 'region']:
        fig = px.histogram(df, x=col, color=col, text_auto=True)
        st.plotly_chart(fig)

    st.markdown("### Charges vs Key Features")
    fig = px.scatter(df, x="age", y="charges", color="smoker", hover_data=['bmi','children'], size='bmi')
    st.plotly_chart(fig)

    fig = px.scatter(df, x="bmi", y="charges", color="smoker", hover_data=['age','children'], size='age')
    st.plotly_chart(fig)

# -----------------------------
# KPI Page
# -----------------------------
elif page == "KPIs":
    st.title("🎯 Key Performance Indicators")
    st.markdown("#### Quick Summary of Insurance Charges")

    total_customers = df.shape[0]
    avg_charges = df['charges'].mean()
    max_charges = df['charges'].max()
    min_charges = df['charges'].min()
    smoker_count = df['smoker'].value_counts().get('yes',0)
    high_bmi_count = df[df['bmi']>30].shape[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", total_customers, help="Total number of customers in dataset")
    col2.metric("Average Charges", f"${avg_charges:,.2f}", help="Mean insurance charges")
    col3.metric("Maximum Charges", f"${max_charges:,.2f}", help="Maximum insurance charges observed")

    col1, col2, col3 = st.columns(3)
    col1.metric("Minimum Charges", f"${min_charges:,.2f}", help="Minimum insurance charges observed")
    col2.metric("Smokers", smoker_count, help="Number of smokers")
    col3.metric("High BMI (>30)", high_bmi_count, help="Number of customers with BMI > 30")

# -----------------------------
# Prediction Page
# -----------------------------
elif page == "Prediction":
    st.title("💵 Predict Insurance Charges")
    st.markdown("Adjust the sliders/inputs below to predict insurance charges for a customer.")

    # Sidebar inputs
    st.sidebar.header("Customer Information")
    age = st.sidebar.slider("Age", int(df['age'].min()), int(df['age'].max()), 30)
    bmi = st.sidebar.slider("BMI", float(df['bmi'].min()), float(df['bmi'].max()), 25.0)
    children = st.sidebar.slider("Number of Children", int(df['children'].min()), int(df['children'].max()), 0)
    sex_male = st.sidebar.selectbox("Sex", [0,1], format_func=lambda x: "Male" if x==1 else "Female")
    smoker_yes = st.sidebar.selectbox("Smoker", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    region_nw = st.sidebar.selectbox("Region - Northwest", [0,1])
    region_se = st.sidebar.selectbox("Region - Southeast", [0,1])
    region_sw = st.sidebar.selectbox("Region - Southwest", [0,1])

    input_df = pd.DataFrame({
        'age':[age],
        'bmi':[bmi],
        'children':[children],
        'sex_male':[sex_male],
        'smoker_yes':[smoker_yes],
        'region_northwest':[region_nw],
        'region_southeast':[region_se],
        'region_southwest':[region_sw],
    })

    input_df['smoker_bmi'] = input_df['smoker_yes'] * input_df['bmi']

    # Add missing dummy columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Scale numeric columns
    num_cols = ['age','bmi','children','smoker_bmi']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    input_df = input_df[feature_columns]

    if st.button("Predict"):
        prediction = xg.predict(input_df)
        st.success(f"Predicted Insurance Charges: ${prediction[0]:,.2f}")

        # -----------------------------
        # SHAP for single input (fixed)
        # -----------------------------
        st.markdown("### 🔍 Feature Impact (SHAP)")

        import shap
        explainer = shap.TreeExplainer(xg)
        shap_values = explainer.shap_values(input_df)

        # Force plotting for Streamlit
        shap.initjs()
        plt.figure(figsize=(8,4))
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        st.pyplot(plt.gcf(), bbox_inches='tight')


# -----------------------------
# Risk Analysis Page
# -----------------------------
elif page == "Risk Analysis":
    st.title("⚠️ Top 10 High-Risk Customers (Charges)")
    top10 = df.sort_values("charges", ascending=False).head(10)
    st.write(top10[['age','bmi','children','sex','smoker','region','charges']])
    fig = px.bar(top10, x='age', y='charges', color='smoker', text='charges', color_discrete_sequence=px.colors.sequential.Reds)
    st.plotly_chart(fig)

# -----------------------------
# Download Page
# -----------------------------
elif page == "Download":
    st.title("📥 Download Dataset")
    csv = df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name="insurance_dataset.csv", mime='text/csv')
