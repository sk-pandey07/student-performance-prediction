import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
data = pd.read_csv("student_data.csv")

# Features
X = data[[
    'attendance',
    'study_hours',
    'previous_marks',
    'sleep_hours',
    'internet_access',
    'tuition',
    'extracurricular',
    'screen_time',
    'assignments_completed'
]]

# Target
y = data['final_marks']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Decision Tree Model

dt_model = DecisionTreeRegressor(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_score = r2_score(y_test, dt_pred)

print("Decision Tree Accuracy:", round(dt_score * 100, 2), "%")

# Random Forest Model

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_score = r2_score(y_test, rf_pred)

print("Random Forest Accuracy:", round(rf_score * 100, 2), "%")

# XGBoost Model

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_score = r2_score(y_test, xgb_pred)

print("XGBoost Accuracy:", round(xgb_score * 100, 2), "%")

# Save Best Model
best_model = xgb_model

joblib.dump(best_model, 'model/student_model.pkl')

print("Best model saved successfully!")

# Attendance vs Final Marks Graph
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=data['attendance'],
    y=data['final_marks']
)

plt.title("Attendance vs Final Marks")
plt.xlabel("Attendance")
plt.ylabel("Final Marks")

plt.show()

# Study Hours vs Final Marks Graph
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=data['study_hours'],
    y=data['final_marks']
)

plt.title("Study Hours vs Final Marks")
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")

plt.show()