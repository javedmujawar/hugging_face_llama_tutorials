import joblib
import numpy as np

# Load the trained model
model = joblib.load("exam_score_predictor.pkl")  # Ensure the file is in the current directory

# Predict exam scores for new study hours
new_study_hours = np.array([[6], [7.5], [9]])  # Example inputs
predicted_scores = model.predict(new_study_hours)

print(predicted_scores)
