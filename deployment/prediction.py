import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('best_model.joblib')  # Single model file

def run():
    # Form for patient data input
    with st.form('form_patient_data'):
        PatientID = st.number_input('Patient ID: ', value=4751, min_value=1, help='Enter Patient ID')
        Age = st.number_input('Age: ', value=65, min_value=15, max_value=100, help='Enter age')
        Gender = st.selectbox('Gender: ', ('0', '1'))
        Ethnicity = st.selectbox('Ethnicity: ', ('0', '1', '2', '3'))
        EducationLevel = st.selectbox('Education Level: ', ('0', '1', '2', '3'))
        BMI = st.number_input('BMI: ', min_value=10.0, max_value=50.0, value=25.0)
        Smoking = st.selectbox('Smoking: ', ('0', '1'))
        AlcoholConsumption = st.slider('Alcohol Consumption: ', min_value=0.0, max_value=20.0, value=5.0)
        PhysicalActivity = st.slider('Physical Activity: ', min_value=0.0, max_value=10.0, value=5.0)
        DietQuality = st.slider('Diet Quality: ', min_value=0.0, max_value=10.0, value=5.0)
        SleepQuality = st.slider('Sleep Quality: ', min_value=0.0, max_value=10.0, value=5.0)
        FamilyHistoryAlzheimers = st.selectbox('Family History of Alzheimer’s: ', ('0', '1'))
        CardiovascularDisease = st.selectbox('Cardiovascular Disease: ', ('0', '1'))
        Diabetes = st.selectbox('Diabetes: ', ('0', '1'))
        Depression = st.selectbox('Depression: ', ('0', '1'))
        HeadInjury = st.selectbox('Head Injury: ', ('0', '1'))
        Hypertension = st.selectbox('Hypertension: ', ('0', '1'))
        SystolicBP = st.number_input('Systolic BP: ', min_value=80, max_value=200, value=120)
        DiastolicBP = st.number_input('Diastolic BP: ', min_value=40, max_value=120, value=80)
        CholesterolTotal = st.number_input('Total Cholesterol: ', min_value=100, max_value=400, value=200)
        CholesterolLDL = st.number_input('LDL Cholesterol: ', min_value=50, max_value=200, value=100)
        CholesterolHDL = st.number_input('HDL Cholesterol: ', min_value=20, max_value=100, value=50)
        CholesterolTriglycerides = st.number_input('Triglycerides: ', min_value=50, max_value=500, value=150)
        
        # New Inputs
        MMSE = st.number_input('MMSE Score: ', min_value=0, max_value=30, value=25)
        FunctionalAssessment = st.selectbox('Functional Assessment: ', ('0', '1'))
        MemoryComplaints = st.selectbox('Memory Complaints: ', ('0', '1'))
        BehavioralProblems = st.selectbox('Behavioral Problems: ', ('0', '1'))
        ADL = st.number_input('Activities of Daily Living (ADL): ', min_value=0.0, max_value=10.0, value=5.0)
        Confusion = st.selectbox('Confusion: ', ('0', '1'))
        Disorientation = st.selectbox('Disorientation: ', ('0', '1'))
        PersonalityChanges = st.selectbox('Personality Changes: ', ('0', '1'))
        DifficultyCompletingTasks = st.selectbox('Difficulty Completing Tasks: ', ('0', '1'))
        Forgetfulness = st.selectbox('Forgetfulness: ', ('0', '1'))

        # Define submit button
        submitted = st.form_submit_button('Predict')

    # Organize data for prediction
    data_inf = {
        'PatientID': PatientID,
        'Age': Age,
        'Gender': Gender,
        'Ethnicity': Ethnicity,
        'EducationLevel': EducationLevel,
        'BMI': BMI,
        'Smoking': Smoking,
        'AlcoholConsumption': AlcoholConsumption,
        'PhysicalActivity': PhysicalActivity,
        'DietQuality': DietQuality,
        'SleepQuality': SleepQuality,
        'FamilyHistoryAlzheimers': FamilyHistoryAlzheimers,
        'CardiovascularDisease': CardiovascularDisease,
        'Diabetes': Diabetes,
        'Depression': Depression,
        'HeadInjury': HeadInjury,
        'Hypertension': Hypertension,
        'SystolicBP': SystolicBP,
        'DiastolicBP': DiastolicBP,
        'CholesterolTotal': CholesterolTotal,
        'CholesterolLDL': CholesterolLDL,
        'CholesterolHDL': CholesterolHDL,
        'CholesterolTriglycerides': CholesterolTriglycerides,
        'MMSE': MMSE,
        'FunctionalAssessment': FunctionalAssessment,
        'MemoryComplaints': MemoryComplaints,
        'BehavioralProblems': BehavioralProblems,
        'ADL': ADL,
        'Confusion': Confusion,
        'Disorientation': Disorientation,
        'PersonalityChanges': PersonalityChanges,
        'DifficultyCompletingTasks': DifficultyCompletingTasks,
        'Forgetfulness': Forgetfulness
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        # If the model internally handles all preprocessing steps, just pass the data directly to the model
        y_pred_inf = model.predict(data_inf)

        # Display prediction
        st.write('## Prediction: ', str(int(y_pred_inf)))

        # Display resources based on prediction
        if y_pred_inf[0] == 0:
            st.write("Since the prediction result is **0**, you may find the following resources helpful:")
            st.markdown("""
            1. [10 Signs of Alzheimer's Disease](https://www.alz.org/alzheimers-dementia/10_signs)
            2. [Alzheimer's Disease Overview - Cleveland Clinic](https://my.clevelandclinic.org/health/diseases/9164-alzheimers-disease)
            """)
        elif y_pred_inf[0] == 1:
            st.write("Since the prediction result is **1**, please check these resources:")
            st.markdown("""
            1. [Care for Alzheimer's Patients - Siloam Hospitals](https://www.siloamhospitals.com/en/informasi-siloam/artikel/perawatan-untuk-pasien-penyakit-alzheimer)
            2. [Caring for People with Alzheimer's - Hello Sehat](https://hellosehat.com/saraf/alzheimer/merawat-orang-dengan-penyakit-alzheimer/)
            3. [How to Care for Parents with Alzheimer's - Kompas](https://www.kompas.id/baca/humaniora/2024/09/09/bagaimana-merawat-orangtua-dengan-alzheimer)
            """)

if __name__ == '__main__':
    run()
