import pandas as pd
import random

data = []

for i in range(2000):

    attendance = random.randint(50, 100)
    study_hours = random.randint(1, 8)
    previous_marks = random.randint(40, 100)
    sleep_hours = random.randint(4, 9)

    internet_access = random.randint(0, 1)
    tuition = random.randint(0, 1)
    extracurricular = random.randint(0, 1)

    screen_time = random.randint(1, 8)
    assignments_completed = random.randint(1, 10)

    final_marks = (
        attendance * 0.2 +
        study_hours * 5 +
        previous_marks * 0.4 +
        sleep_hours * 2 +
        internet_access * 5 +
        tuition * 5 +
        extracurricular * 3 -
        screen_time * 1.5 +
        assignments_completed * 2
    )

    final_marks = round(min(final_marks, 100), 2)

    data.append([
        attendance,
        study_hours,
        previous_marks,
        sleep_hours,
        internet_access,
        tuition,
        extracurricular,
        screen_time,
        assignments_completed,
        final_marks
    ])

columns = [
    'attendance',
    'study_hours',
    'previous_marks',
    'sleep_hours',
    'internet_access',
    'tuition',
    'extracurricular',
    'screen_time',
    'assignments_completed',
    'final_marks'
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("student_data.csv", index=False)

print("2000 rows dataset generated successfully!")