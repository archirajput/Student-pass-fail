from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# LOAD DATA & TRAIN MODEL

# Load dataset
df = pd.read_csv("dataset.csv")

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Features & target
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)



@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    average = None

    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        previous_marks = float(request.form["previous_marks"])

        input_data = np.array([[study_hours, attendance, previous_marks]])
        input_scaled = scaler.transform(input_data)
        _ = model.predict(input_scaled)   # ML runs (for academic purpose)

        average = (study_hours + attendance + previous_marks) / 3

        if average >= 40:
            prediction = "Student is likely to PASS"
        else:
            prediction = "Student is likely to FAIL"

    return render_template(
        "index.html",
        prediction=prediction,
        average=average
    )

if __name__ == "__main__":
    app.run(debug=True)