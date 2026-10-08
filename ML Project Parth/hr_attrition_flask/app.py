from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        data = {
            "Age": int(request.form["Age"]),
            "DailyRate": int(request.form["DailyRate"]),
            "DistanceFromHome": int(request.form["DistanceFromHome"]),
            "Education": int(request.form["Education"]),
            "EmployeeCount": int(request.form["EmployeeCount"]),
            "EmployeeNumber": int(request.form["EmployeeNumber"]),
            "EnvironmentSatisfaction": int(request.form["EnvironmentSatisfaction"]),
            "HourlyRate": int(request.form["HourlyRate"]),
            "JobInvolvement": int(request.form["JobInvolvement"]),
            "JobLevel": int(request.form["JobLevel"]),
            "JobSatisfaction": int(request.form["JobSatisfaction"]),
            "MonthlyIncome": int(request.form["MonthlyIncome"]),
            "MonthlyRate": int(request.form["MonthlyRate"]),
            "NumCompaniesWorked": int(request.form["NumCompaniesWorked"]),
            "PercentSalaryHike": int(request.form["PercentSalaryHike"]),
            "PerformanceRating": int(request.form["PerformanceRating"]),
            "RelationshipSatisfaction": int(request.form["RelationshipSatisfaction"]),
            "StandardHours": int(request.form["StandardHours"]),
            "StockOptionLevel": int(request.form["StockOptionLevel"]),
            "TotalWorkingYears": int(request.form["TotalWorkingYears"]),
            "TrainingTimesLastYear": int(request.form["TrainingTimesLastYear"]),
            "WorkLifeBalance": int(request.form["WorkLifeBalance"]),
            "YearsAtCompany": int(request.form["YearsAtCompany"]),
            "YearsInCurrentRole": int(request.form["YearsInCurrentRole"]),
            "YearsSinceLastPromotion": int(request.form["YearsSinceLastPromotion"]),
            "YearsWithCurrManager": int(request.form["YearsWithCurrManager"]),
            "BusinessTravel_Travel_Frequently": 1 if request.form["BusinessTravel"] == "Travel_Frequently" else 0,
            "BusinessTravel_Travel_Rarely": 1 if request.form["BusinessTravel"] == "Travel_Rarely" else 0,
            "Department_Research & Development": 1 if request.form["Department"] == "Research & Development" else 0,
            "Department_Sales": 1 if request.form["Department"] == "Sales" else 0,
            "EducationField_Life Sciences": 1 if request.form["EducationField"] == "Life Sciences" else 0,
            "EducationField_Marketing": 1 if request.form["EducationField"] == "Marketing" else 0,
            "EducationField_Medical": 1 if request.form["EducationField"] == "Medical" else 0,
            "EducationField_Other": 1 if request.form["EducationField"] == "Other" else 0,
            "EducationField_Technical Degree": 1 if request.form["EducationField"] == "Technical Degree" else 0,
            "Gender_Male": 1 if request.form["Gender"] == "Male" else 0,
            "JobRole_Human Resources": 1 if request.form["JobRole"] == "Human Resources" else 0,
            "JobRole_Laboratory Technician": 1 if request.form["JobRole"] == "Laboratory Technician" else 0,
            "JobRole_Manager": 1 if request.form["JobRole"] == "Manager" else 0,
            "JobRole_Manufacturing Director": 1 if request.form["JobRole"] == "Manufacturing Director" else 0,
            "JobRole_Research Director": 1 if request.form["JobRole"] == "Research Director" else 0,
            "JobRole_Research Scientist": 1 if request.form["JobRole"] == "Research Scientist" else 0,
            "JobRole_Sales Executive": 1 if request.form["JobRole"] == "Sales Executive" else 0,
            "JobRole_Sales Representative": 1 if request.form["JobRole"] == "Sales Representative" else 0,
            "MaritalStatus_Married": 1 if request.form["MaritalStatus"] == "Married" else 0,
            "MaritalStatus_Single": 1 if request.form["MaritalStatus"] == "Single" else 0,
            "OverTime_Yes": 1 if request.form["OverTime"] == "Yes" else 0
        }

        columns = list(data.keys())
        employee = pd.DataFrame([data], columns=columns)
        employee = scaler.transform(employee)

        prediction = model.predict(employee)[0]
        probability = model.predict_proba(employee)[0][1] * 100

        if prediction == 1:
            prediction_text = "Leave"
        else:
            prediction_text = "Stay"

        if probability <= 30:
            risk = "Low"
        elif probability <= 60:
            risk = "Medium"
        else:
            risk = "High"

        result = {
            "prediction": prediction_text,
            "probability": f"{probability:.2f}",
            "risk": risk
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
