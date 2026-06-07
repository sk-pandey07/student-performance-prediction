from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('model/student_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Input values
    attendance = float(request.form['attendance'])
    study_hours = float(request.form['study_hours'])
    previous_marks = float(request.form['previous_marks'])
    sleep_hours = float(request.form['sleep_hours'])

    internet_access = int(request.form['internet_access'])
    tuition = int(request.form['tuition'])
    extracurricular = int(request.form['extracurricular'])

    screen_time = float(request.form['screen_time'])
    assignments_completed = int(request.form['assignments_completed'])

    # Feature array
    features = np.array([[
        attendance,
        study_hours,
        previous_marks,
        sleep_hours,
        internet_access,
        tuition,
        extracurricular,
        screen_time,
        assignments_completed
    ]])

    # Prediction
    prediction = model.predict(features)[0]

    # Limit prediction between 0 and 100
    prediction = max(0, min(prediction, 100))

    # Grade Logic
    if prediction >= 90:
        grade = "A+"
    elif prediction >= 75:
        grade = "A"
    elif prediction >= 60:
        grade = "B"
    elif prediction >= 40:
        grade = "C"
    else:
        grade = "Fail"

    # Suggestions
    suggestions = []

    if attendance < 75:
        suggestions.append("Improve attendance.")

    if study_hours < 3:
        suggestions.append("Increase daily study hours.")

    if sleep_hours < 6:
        suggestions.append("Take proper sleep.")

    if screen_time > 6:
        suggestions.append("Reduce screen time.")

    if assignments_completed < 5:
        suggestions.append("Complete more assignments.")

    return render_template(
        'result.html',
        prediction=round(prediction, 2),
        grade=grade,
        suggestions=suggestions
    )


if __name__ == '__main__':
    app.run(debug=True)