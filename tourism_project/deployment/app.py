import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Wellness Tourism Package Prediction", page_icon="🧘", layout="centered")

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_model.joblib")   # filename of the trained model saved by train.py
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("🧘 Wellness Tourism Package - Purchase Prediction")
st.caption(
    "Visit with Us | Predicts whether a customer is likely to purchase the "
    "newly-introduced Wellness Tourism Package, before the sales team calls them."
)
st.write("Fill in the customer details below to get a prediction, purchase probability, recommendation and confidence level.")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.slider("Age", 18, 70, 30)
        TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        CityTier = st.selectbox("City Tier", [1, 2, 3])
        DurationOfPitch = st.slider("Duration of Pitch (mins)", 0, 100, 15)
        Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
        NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 3)
        ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

    with col2:
        PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
        MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
        NumberOfTrips = st.slider("Number of Trips (per year)", 1, 20, 3)
        Passport = st.selectbox("Has Passport?", ["Yes", "No"])
        PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
        NumberOfChildrenVisiting = st.slider("Number of Children Visiting (below age 5)", 0, 5, 1)
        Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=30000.0, step=500.0)

    submitted = st.form_submit_button("Predict")

# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Classification threshold used consistently with model_building/train.py
CLASSIFICATION_THRESHOLD = 0.45

if submitted:
    prob = float(model.predict_proba(input_data)[0, 1])
    pred = int(prob >= CLASSIFICATION_THRESHOLD)

    if prob >= 0.7:
        confidence = "High"
    elif prob >= 0.45:
        confidence = "Medium"
    else:
        confidence = "Low"

    st.divider()
    if pred == 1:
        st.success(f"### Prediction: Likely to PURCHASE the Wellness Package")
        recommendation = "Prioritize this customer for outreach and pitch the Wellness Tourism Package directly."
    else:
        st.warning(f"### Prediction: Unlikely to purchase")
        recommendation = "Deprioritize for the Wellness Tourism Package, or nurture with a lower-commitment offer first."

    m1, m2, m3 = st.columns(3)
    m1.metric("Purchase Probability", f"{prob:.1%}")
    m2.metric("Confidence Level", confidence)
    m3.metric("Predicted Class", "Purchase" if pred == 1 else "No Purchase")

    st.write(f"**Recommendation:** {recommendation}")
    st.caption(f"Classification threshold: {CLASSIFICATION_THRESHOLD} (tuned to favor recall on likely buyers)")
