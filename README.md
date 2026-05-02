# 📊 AI Customer Churn Predictor & Dashboard

An end-to-end Data Science project featuring a machine learning pipeline and an interactive web dashboard. This application predicts the likelihood of customer churn (cancellation) using a Random Forest classifier.

## 🚀 Project Overview
In this project, we address the business challenge of customer retention. By analyzing customer behavior—such as tenure, monthly spending, and support interactions—the model identifies high-risk customers, allowing businesses to take proactive retention measures.

### Key Features
*   **Machine Learning Pipeline:** Data preprocessing, training, and evaluation using `Scikit-Learn`.
*   **Interactive Dashboard:** A real-time web UI built with `Streamlit`.
*   **Predictive Simulation:** Sidebar sliders allow users to input custom data to see "what-if" scenarios.
*   **Data Visualization:** Interactive charts using `Plotly` to show feature importance and data distributions.

## 🛠️ Tech Stack
*   **Language:** Python 3.x
*   **Libraries:** Pandas, NumPy, Scikit-Learn
*   **Visualization:** Plotly, Seaborn, Matplotlib
*   **Deployment:** Streamlit

## 📈 The Model
We utilized a **Random Forest Classifier** consisting of 100 decision trees. 
*   **Objective:** Binary Classification (Churn vs. Stay).
*   **Evaluation:** The model is evaluated based on its Accuracy, Precision, and Recall via a confusion matrix.
