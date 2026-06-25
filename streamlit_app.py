import streamlit as st
import pandas as pd
import joblib
import pickle

st.set_page_config(page_title="Phishing URL Detector", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('logistic_regression_model.pkl')
    with open('shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    feature_names = list(pd.read_csv('X_train.csv').columns)
    return model, explainer, feature_names

model, explainer, feature_names = load_artifacts()

st.title("🔍 Phishing Website Detection")
st.write("Upload a CSV of feature values, or pick a sample row, to get a prediction with explanation.")

mode = st.radio("Input method", ["Upload CSV", "Pick a test sample"])
input_df = None

if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV with feature columns", type="csv")
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        st.write("Preview:", input_df.head())
else:
    test_df = pd.read_csv('X_test.csv')
    row_idx = st.number_input("Pick a test row index", 0, len(test_df)-1, 0)
    input_df = test_df.iloc[[row_idx]]
    st.write("Selected row:", input_df)

if input_df is not None and st.button("Predict"):
    X = input_df[feature_names]
    preds = model.predict(X)
    probs = model.predict_proba(X)

    for i in range(len(X)):
        st.subheader(f"Row {i}")
        label = "🔴 Phishing" if preds[i] == 1 else "🟢 Legitimate"
        st.metric("Prediction", label, f"Confidence: {max(probs[i]):.2%}")

        shap_vals = explainer.shap_values(X.iloc[[i]])
        top5 = sorted(zip(feature_names, shap_vals[0]), key=lambda x: abs(x[1]), reverse=True)[:5]
        st.write("**Top contributing features:**")
        for feat, val in top5:
            direction = "increases" if val > 0 else "decreases"
            st.write(f"- `{feat}` {direction} phishing likelihood (SHAP value: {val:.4f})")
