import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Predictor", page_icon="👥", layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    return model, scaler


model, scaler = load_artifacts()

st.title("👥 Employee Attrition Predictor")

with st.form("employee_form"):
    st.subheader("Personal Details")
    c1, c2, c3 = st.columns(3)
    Age = c1.number_input("Age", 18, 70, 30)
    Gender = c2.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = c3.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    DistanceFromHome = c1.number_input("Distance From Home (km)", 1, 50, 5)
    Education = c2.selectbox("Education (1-5)", [1, 2, 3, 4, 5], index=2)
    EducationField = c3.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other",
        ],
    )

    st.subheader("Job Details")
    c1, c2, c3 = st.columns(3)
    Department = c1.selectbox(
        "Department", ["Research & Development", "Sales", "Human Resources"]
    )
    JobRole = c2.selectbox(
        "Job Role",
        [
            "Healthcare Representative",
            "Human Resources",
            "Laboratory Technician",
            "Manager",
            "Manufacturing Director",
            "Research Director",
            "Research Scientist",
            "Sales Executive",
            "Sales Representative",
        ],
    )
    BusinessTravel = c3.selectbox(
        "Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    )
    JobLevel = c1.selectbox("Job Level (1-5)", [1, 2, 3, 4, 5])
    OverTime = c2.selectbox("OverTime", ["No", "Yes"])
    StockOptionLevel = c3.selectbox("Stock Option Level (0-3)", [0, 1, 2, 3])

    st.subheader("Compensation")
    c1, c2, c3 = st.columns(3)
    MonthlyIncome = c1.number_input("Monthly Income", 1000, 25000, 5000, step=100)
    DailyRate = c2.number_input("Daily Rate", 100, 1500, 800)
    HourlyRate = c3.number_input("Hourly Rate", 30, 100, 65)
    MonthlyRate = c1.number_input("Monthly Rate", 2000, 27000, 14000, step=100)
    PercentSalaryHike = c2.number_input("Percent Salary Hike", 0, 30, 15)
    PerformanceRating = c3.selectbox("Performance Rating (1-4)", [1, 2, 3, 4], index=2)

    st.subheader("Experience")
    c1, c2, c3 = st.columns(3)
    TotalWorkingYears = c1.number_input("Total Working Years", 0, 50, 8)
    NumCompaniesWorked = c2.number_input("Num Companies Worked", 0, 15, 2)
    TrainingTimesLastYear = c3.number_input("Training Times Last Year", 0, 10, 3)
    YearsAtCompany = c1.number_input("Years At Company", 0, 50, 5)
    YearsInCurrentRole = c2.number_input("Years In Current Role", 0, 50, 3)
    YearsSinceLastPromotion = c3.number_input("Years Since Last Promotion", 0, 30, 1)
    YearsWithCurrManager = c1.number_input("Years With Current Manager", 0, 30, 3)

    st.subheader("Satisfaction & Involvement (1-4)")
    c1, c2, c3 = st.columns(3)
    EnvironmentSatisfaction = c1.selectbox(
        "Environment Satisfaction", [1, 2, 3, 4], index=2
    )
    JobSatisfaction = c2.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
    RelationshipSatisfaction = c3.selectbox(
        "Relationship Satisfaction", [1, 2, 3, 4], index=2
    )
    JobInvolvement = c1.selectbox("Job Involvement", [1, 2, 3, 4], index=2)
    WorkLifeBalance = c2.selectbox("Work Life Balance", [1, 2, 3, 4], index=2)

    with st.expander("Other fields (usually constant in the dataset)"):
        c1, c2, c3 = st.columns(3)
        EmployeeCount = c1.number_input("Employee Count", 1, 1, 1)
        StandardHours = c2.number_input("Standard Hours", 80, 80, 80)
        EmployeeNumber = c3.number_input("Employee Number", 1, 100000, 1)

    submitted = st.form_submit_button(
        "Predict", type="primary", use_container_width=True
    )

if submitted:
    data = {
        "Age": Age,
        "DailyRate": DailyRate,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EmployeeCount": EmployeeCount,
        "EmployeeNumber": EmployeeNumber,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobSatisfaction": JobSatisfaction,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StandardHours": StandardHours,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
        "BusinessTravel_Travel_Frequently": int(BusinessTravel == "Travel_Frequently"),
        "BusinessTravel_Travel_Rarely": int(BusinessTravel == "Travel_Rarely"),
        "Department_Research & Development": int(
            Department == "Research & Development"
        ),
        "Department_Sales": int(Department == "Sales"),
        "EducationField_Life Sciences": int(EducationField == "Life Sciences"),
        "EducationField_Marketing": int(EducationField == "Marketing"),
        "EducationField_Medical": int(EducationField == "Medical"),
        "EducationField_Other": int(EducationField == "Other"),
        "EducationField_Technical Degree": int(EducationField == "Technical Degree"),
        "Gender_Male": int(Gender == "Male"),
        "JobRole_Human Resources": int(JobRole == "Human Resources"),
        "JobRole_Laboratory Technician": int(JobRole == "Laboratory Technician"),
        "JobRole_Manager": int(JobRole == "Manager"),
        "JobRole_Manufacturing Director": int(JobRole == "Manufacturing Director"),
        "JobRole_Research Director": int(JobRole == "Research Director"),
        "JobRole_Research Scientist": int(JobRole == "Research Scientist"),
        "JobRole_Sales Executive": int(JobRole == "Sales Executive"),
        "JobRole_Sales Representative": int(JobRole == "Sales Representative"),
        "MaritalStatus_Married": int(MaritalStatus == "Married"),
        "MaritalStatus_Single": int(MaritalStatus == "Single"),
        "OverTime_Yes": int(OverTime == "Yes"),
    }

    employee = pd.DataFrame([data], columns=list(data.keys()))
    employee_scaled = scaler.transform(employee)

    prediction = model.predict(employee_scaled)[0]
    probability = model.predict_proba(employee_scaled)[0][1] * 100

    prediction_text = "Leave" if prediction == 1 else "Stay"

    if probability <= 30:
        risk = "Low"
    elif probability <= 60:
        risk = "Medium"
    else:
        risk = "High"

    st.divider()
    st.subheader("Result")
    r1, r2, r3 = st.columns(3)
    r1.metric("Prediction", prediction_text)
    r2.metric("Leave Probability", f"{probability:.2f}%")
    r3.metric("Risk Level", risk)
    st.progress(min(int(probability), 100))

    if risk == "High":
        st.error("High attrition risk — this employee is likely to leave.")
    elif risk == "Medium":
        st.warning("Medium attrition risk — worth keeping an eye on.")
    else:
        st.success("Low attrition risk — this employee is likely to stay.")
