import os
import pickle
import numpy as np
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# 2. Cache Model Loading for Production Efficiency
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file '{model_path}' not found! Please place it in the same directory.")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# 3. Application UI
st.title("🏠 House Price Prediction App")
st.write("Enter the property details below to get an instant valuation estimate.")

if model is not None:
    # Check if the model has feature names tracked
    features = getattr(model, "feature_names_in_", ["area", "bedrooms", "bathrooms"])
    
    st.markdown("### 📋 Property Specifications")
    
    # Input controls with reasonable bounds
    col1, col2, col3 = st.columns(3)
    
    with col1:
        area = st.number_input(
            "Square Footage (sqft)", 
            min_value=100, 
            max_value=10000, 
            value=1500, 
            step=50
        )
    with col2:
        bedrooms = st.slider("Bedrooms", min_value=1, max_value=8, value=3)
    with col3:
        bathrooms = st.slider("Bathrooms", min_value=1, max_value=6, value=2)
        
    st.markdown("---")
    
    # 4. Prediction Logic
    # Shape the input array exactly how sklearn expects it: (1, n_features)
    input_data = np.array([[area, bedrooms, bathrooms]])
    
    if st.button("💰 Estimate Price", type="primary"):
        with st.spinner("Calculating estimate..."):
            try:
                prediction = model.predict(input_data)[0]
                
                # Render prediction formatted cleanly as currency
                st.success(f"### Estimated Value: ${prediction:,.2f}")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Optional: Display Model Parameters (Good for transparency)
    with st.expander("🛠️ View Model Coefficients & Formula"):
        st.write(f"**Intercept:** {model.intercept_:.4f}")
        st.write(f"**Coefficients:**")
        for name, coef in zip(features, model.coef_):
            st.write(f"- {name}: {coef:.4f}")
