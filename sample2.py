import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from huggingface_hub import login, Repository

# 1. Train the model
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
exam_scores = np.array([50, 55, 60, 65, 70, 75, 78, 85, 90, 95])

model = LinearRegression()
model.fit(study_hours, exam_scores)

# 2. Save the model
model_filename = "exam_score_predictor.pkl"
joblib.dump(model, model_filename)

# 3. Log in to Hugging Face
login(token="")  # Use your actual Hugging Face token

# 4. Define repo details
repo_name = "exam-score-predictor_test_javed1"  
repo_id = "mujawarjaved/exam-score-predictor_test_javed"  # Change to your actual repo name

# 5. Clone the existing repository (skip create_repo)
repo = Repository(local_dir=repo_name, clone_from=f"https://huggingface.co/{repo_id}")

# 6. Move the model file to the repo folder
os.rename(model_filename, f"{repo_name}/{model_filename}")

# 7. Push to Hugging Face
repo.push_to_hub(commit_message="Updated model test")
