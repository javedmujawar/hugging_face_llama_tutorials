from huggingface_hub import hf_hub_download
import joblib
import numpy as np

# Define repository details
repo_id = "mujawarjaved/exam-score-predictor_test_javed"  # Your Hugging Face repo
model_filename = "exam_score_predictor.pkl"

# Download the model file from Hugging Face
model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)

# Load the model
model = joblib.load(model_path)

# Predict exam scores
new_study_hours = np.array([[6], [7.5], [9]])  # Example inputs
predicted_scores = model.predict(new_study_hours)

print(predicted_scores)
