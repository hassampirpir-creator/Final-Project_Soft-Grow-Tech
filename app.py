import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Customer Churn Predictor", layout="wide")


@st.cache_data
def load_data():
    np.random.seed(42)
    data_size = 2000
    df = pd.DataFrame({
        'Tenure_Months': np.random.randint(1, 72, data_size),
        'Monthly_Charges': np.random.uniform(20, 120, data_size),
        'Total_Data_GB': np.random.uniform(10, 500, data_size),
        'Support_Calls': np.random.randint(0, 10, data_size),
        'Churn': np.random.choice([0, 1], size=data_size, p=[0.7, 0.3])
    })
    return df

df = load_data()


X = df.drop('Churn', axis=1)
y = df['Churn']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


st.sidebar.header("🔍 Input Customer Details")
def user_input_features():
    tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
    charges = st.sidebar.slider("Monthly Charges ($)", 20, 120, 50)
    data_gb = st.sidebar.slider("Data Usage (GB)", 10, 500, 100)
    calls = st.sidebar.slider("Support Calls", 0, 10, 2)
    
    features = pd.DataFrame([[tenure, charges, data_gb, calls]], 
                            columns=['Tenure_Months', 'Monthly_Charges', 'Total_Data_GB', 'Support_Calls'])
    return features

input_df = user_input_features()


st.title("📊 Data Science Project: Churn Prediction")
st.markdown("""
This app uses a **Random Forest Machine Learning model** to predict whether a customer is likely to leave your service.
""")


col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Probability")
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    if prediction[0] == 1:
        st.error(f"High Risk: {prediction_proba[0][1]:.2%} probability of Churning")
    else:
        st.success(f"Low Risk: {prediction_proba[0][0]:.2%} probability of Staying")

with col2:
    st.subheader("Feature Importance")
    importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    fig = px.bar(importances, x='Importance', y='Feature', orientation='h', color='Importance')
    st.plotly_chart(fig, use_container_width=True)


st.divider()
st.subheader("Exploratory Data Analysis")
chart_choice = st.selectbox("Select View", ["Charges vs Churn", "Support Calls vs Churn"])

if chart_choice == "Charges vs Churn":
    fig_eda = px.box(df, x="Churn", y="Monthly_Charges", color="Churn", title="Charges Distribution")
    st.plotly_chart(fig_eda)
else:
    fig_eda = px.histogram(df, x="Support_Calls", color="Churn", barmode="group", title="Calls vs Churn Frequency")
    st.plotly_chart(fig_eda)

st.write("Current Dataset Sample:", df.head())