import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_models():
    try:
        prep = joblib.load('preprocessor.joblib')
        mod = joblib.load('best_model.joblib')
        return prep, mod
    except FileNotFoundError:
        st.error("Models not found! Please ensure 'preprocessor.joblib' and 'best_model.joblib' exist.")
        st.stop()

preprocessor, model = load_models()

st.set_page_config(page_title="Autism Prediction App", page_icon="🧠", layout="centered")

st.markdown(
    """
    <style>
    /* Target the label of the radio buttons (the questions) */
    div[data-testid="stRadio"] > label {
        font-size: 15px !important;
        font-weight: bold !important;
    }
    
    div[data-testid="stRadio"] > label p {
        font-size: 15px !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Autism Prediction App")
st.markdown("Please enter the patient's behavioral screening scores and demographic details below.")


st.subheader("1. Screening Questions (A1 - A10)")
st.markdown("(1 = Yes / Observed, 0 = No / Not Observed)")

questions = [
    "A1: I often notice small sounds when others do not.",
    "A2: I usually concentrate more on the whole picture, rather than the small details.",
    "A3: I find it easy to do more than one thing at once.",
    "A4: If there is an interruption, I can switch back to what I was doing very quickly.",
    "A5: I find it easy to 'read between the lines' when someone is talking to me.",
    "A6: I know how to tell if someone listening to me is getting bored.",
    "A7: When I'm reading a story, I find it difficult to work out the characters' intentions.",
    "A8: I like to collect information about categories of things.",
    "A9: I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    "A10: I find it difficult to work out people's intentions."
]

scores = {}
for i, q in enumerate(questions, 1):
    scores[f'A{i}_Score'] = st.radio(q, [0, 1], index=0, horizontal=True)


st.subheader("2. Demographic Information")

col_d1, col_d2 = st.columns(2)

with col_d1:
    age = st.number_input("Age (Years)", min_value=1.0, max_value=120.0, value=25.0, step=1.0)
    gender_input = st.selectbox("Gender", ["Male", "Female"])
    ethnicity_input = st.selectbox(
        "Ethnicity", 
        ["White-European", "Latino", "Black", "Asian", "Middle Eastern ", "Pasifika", "South Asian", "Hispanic", "Turkish", "Others"]
    )
    relation_input = st.selectbox(
        "Who is completing the test?", 
        ["Self", "Parent", "Health care professional", "Relative", "Others"]
    )

with col_d2:
    jaundice_input = st.selectbox("Born with Jaundice?", ["No", "Yes"])
    used_app_input = st.selectbox("Used this screening app before?", ["No", "Yes"])
    
    # Using the exact countries mapped from the notebook (counts >= 10)
    valid_countries = [
        "United States", "India", "New Zealand", "United Kingdom", "Jordan", 
        "United Arab Emirates", "Australia", "Canada", "Afghanistan", "Netherlands", 
        "Austria", "Sri Lanka", "Brazil", "France", "Kazakhstan", "Spain", "Other"
    ]
    country_input = st.selectbox("Country of Residence", valid_countries)


st.markdown("---")
if st.button("Calculate Prediction", type="primary"):
    
   
    gender_mapped = 'm' if gender_input == "Male" else 'f'
    jaundice_mapped = 1 if jaundice_input == "Yes" else 0
    used_app_mapped = 1 if used_app_input == "Yes" else 0


    user_data = {
        'A1_Score': scores['A1_Score'],
        'A2_Score': scores['A2_Score'],
        'A3_Score': scores['A3_Score'],
        'A4_Score': scores['A4_Score'],
        'A5_Score': scores['A5_Score'],
        'A6_Score': scores['A6_Score'],
        'A7_Score': scores['A7_Score'],
        'A8_Score': scores['A8_Score'],
        'A9_Score': scores['A9_Score'],
        'A10_Score': scores['A10_Score'],
        'age': age,
        'gender': gender_mapped,
        'ethnicity': ethnicity_input,
        'jaundice': jaundice_mapped,
        'contry_of_res': country_input,
        'used_app_before': used_app_mapped,
        'relation': relation_input
    }

    
    input_df = pd.DataFrame([user_data])
    input_df = input_df.reindex(columns=preprocessor.feature_names_in_)

    try:
        
        processed_input = preprocessor.transform(input_df)
        
        
        prediction = model.predict(processed_input)[0]

       
        st.subheader("Result")
        
        if prediction == 1:
            st.error(f"**High Risk of Autism.**")
        else:
            st.success(f" **Low Risk of Autism.**")
            
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")