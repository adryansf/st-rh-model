# importar bibliotecas
import streamlit as st
import pandas as pd
import pydeck as pdk
import pickle
import keras
import tensorflow
import sklearn

# Importar modelo
model = pickle.load(open("model.pickle", 'rb'))
scaler = pickle.load(open("scaler.pickle", 'rb'))

# Transformações
education_transform = {
  "Below College": 0,
  "College": 1,
  "Bachelor": 2,
  "Master": 3,
  "Doctor": 4
}

levels = {
  'Low': 1,
  'Medium': 2,
  'High': 3,
  'Very High': 4,
}

quality = {
  'Bad': 1,
  'Good': 2,
  'Better': 3,
  'Best': 4,
}

boolean = {
  'Yes': 1,
  'No': 0
}

def test_one_hot_encode(t, v):
  return 1.0 if t == v else 0

# SIDEBAR
st.sidebar.header("Parâmetros")

# Age
age = st.sidebar.slider('Age', min_value=18, max_value=100)

# Gender
gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])

# DistanceFromHome
distance_from_home = st.sidebar.number_input('Distance From Home', step=1, min_value=0)

# Education
education = st.sidebar.selectbox('Education', ["Below College","College","Bachelor","Master","Doctor"])

# EducationField
education_field = st.sidebar.selectbox('Education Field', ['Life Sciences','Other','Medical', 'Marketing', 'Technical Degree', 
  'Human Resources'])

# Environment Satisfaction
environment_satisfaction = st.sidebar.selectbox('Environment Satisfaction', ["Low", "Medium", "High", "Very High"])

# JobInvolvement
job_involvement = st.sidebar.selectbox('Job Involvement', ["Low", "Medium", "High", "Very High"])

# Job Level
job_level = st.sidebar.slider('Job Level', min_value=1, max_value=5)

# Job Role
job_role = st.sidebar.selectbox('Job Level', ['Sales Executive','Research Scientist','Laboratory Technician',
  'Manufacturing Director','Healthcare Representative','Manager','Sales Representative' 'Research Director',
  'Human Resources'])

# JobSatisfaction
job_satisfaction = st.sidebar.selectbox('Job Satisfaction', ["Low", "Medium", "High", "Very High"])

# MonthlyIncome
monthly_income = st.sidebar.number_input('Monthly Income ($)', min_value=0, step=1)

# NumCompaniesWorked
num_companies_worked = st.sidebar.number_input('Nº Companies Worked', min_value=0, step=1)

# Relationship Satisfaction
relationship_satisfaction = st.sidebar.selectbox('Relationship Satisfaction', ["Low", "Medium", "High", "Very High"])

# StockOptionLevel
stock_option_level = st.sidebar.slider('Stock Option Level', min_value = 0, max_value = 3)

# TotalWorkingYears
total_working_years = st.sidebar.number_input('Total Working Years', min_value = 0, step = 1)

# TrainingTimesLastYear
training_times_last_year = st.sidebar.number_input('Training Times Last Year', min_value = 0, step = 1)

# WorkLifeBalance
work_life_balance = st.sidebar.selectbox('Work Life Balance', ["Bad","Good","Better","Best"])

# YearsAtCompany
years_at_company = st.sidebar.number_input('Years At Company', min_value = 0, step = 1)

# YearsInCurrentRole
years_in_current_role = st.sidebar.number_input('Years In Current Role', min_value = 0, step = 1)

# YearsSinceLastPromotion
years_since_last_promotion = st.sidebar.number_input('Years Since Last Promotion', min_value = 0, step = 1)

# YearsWithCurrManager
years_with_current_manager = st.sidebar.number_input('Years With Current Manager', min_value = 0, step = 1)

# MaritalStatus
marital_status = st.sidebar.selectbox('Marital Status', ["Single","Married","Divorced"])

# Department
departament = st.sidebar.selectbox('Departament', ['Sales', 'Research & Development', 'Human Resources'])

# BusinessTravel
business_travel = st.sidebar.selectbox('Business Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])

# OverTime
overtime = st.sidebar.selectbox('Over Time', ['No', 'Yes'])

# Definindo DataFrame
data = {
  'Age': [age],
  'DistanceFromHome': [distance_from_home],
  'Education': [education_transform[education]],
  'EnvironmentSatisfaction': [levels[environment_satisfaction]],
  'Gender': [1 if gender == 'Male' else 0],
  'JobInvolvement': [levels[job_involvement]],
  'JobLevel': [job_level],
  'JobSatisfaction': [levels[job_satisfaction]],  
  'MonthlyIncome': [monthly_income],
  'NumCompaniesWorked': [num_companies_worked],
  'OverTime': [boolean[overtime]],
  'RelationshipSatisfaction': [levels[relationship_satisfaction]],
  'StockOptionLevel': [stock_option_level],
  'TotalWorkingYears': [total_working_years],
  'TrainingTimesLastYear': [training_times_last_year],
  'WorkLifeBalance': [quality[work_life_balance]],
  'YearsAtCompany': [years_at_company],
  'YearsInCurrentRole': [years_in_current_role],
  'YearsSinceLastPromotion': [years_since_last_promotion],
  'YearsWithCurrManager': [years_with_current_manager],
  'MaritalStatus_Divorced': [test_one_hot_encode('Divorced', marital_status)],
  'MaritalStatus_Married': [test_one_hot_encode('Married', marital_status)],
  'MaritalStatus_Single': [test_one_hot_encode('Single', marital_status)],
  'JobRole_Healthcare Representative': [test_one_hot_encode('Healthcare Representative', job_role)],
  'JobRole_Human Resources': [test_one_hot_encode('Human Resources', job_role)], 
  'JobRole_Laboratory Technician': [test_one_hot_encode('Laboratory Technician', job_role)],
  'JobRole_Manager': [test_one_hot_encode('Manager', job_role)],
  'JobRole_Manufacturing Director': [test_one_hot_encode('Manufacturing Director', job_role)],
  'JobRole_Research Director': [test_one_hot_encode('Research Director', job_role)], 
  'JobRole_Research Scientist': [test_one_hot_encode('Research Scientist', job_role)],
  'JobRole_Sales Executive': [test_one_hot_encode('Sales Executive', job_role)], 
  'JobRole_Sales Representative': [test_one_hot_encode('Sales Representative', job_role)],
  'EducationField_Human Resources': [test_one_hot_encode('Human Resources', education_field)],
  'EducationField_Life Sciences': [test_one_hot_encode('Life Sciences', education_field)],
  'EducationField_Marketing': [test_one_hot_encode('Marketing', education_field)],
  'EducationField_Medical': [test_one_hot_encode('Medical', education_field)],
  'EducationField_Other': [test_one_hot_encode('Other', education_field)],
  'EducationField_Technical Degree': [test_one_hot_encode('Technical Degree', education_field)],
  'Department_Human Resources': [test_one_hot_encode('Human Resources', departament)],
  'Department_Research & Development': [test_one_hot_encode('Research & Development', departament)],
  'Department_Sales': [test_one_hot_encode('Sales', departament)], 
  'BusinessTravel_Non-Travel': [test_one_hot_encode('Non-Travel', business_travel)],
  'BusinessTravel_Travel_Frequently': [test_one_hot_encode('Travel_Frequently', business_travel)],
  'BusinessTravel_Travel_Rarely': [test_one_hot_encode('Travel_Rarely', business_travel)],
}

df = pd.DataFrame(data)
X = scaler.transform(df)

result = "irá deixar a empresa" if model.predict(X) > 0.5 else "não deixará a empresa"



# MAIN
st.title("Departamento de Recursos Humanos")
st.dataframe(df)
st.markdown("Resultado: O Funcionário :red["+ result +"].")