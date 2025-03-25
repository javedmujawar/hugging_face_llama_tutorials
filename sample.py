import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from huggingface_hub import login, create_repo, Repository

# 1. Train the model
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
exam_scores = np.array([50, 55, 60, 65, 70, 75, 78, 85, 90, 95])

model = LinearRegression()
model.fit(study_hours, exam_scores)

# 2. Save the model
joblib.dump(model, "exam_score_predictor.pkl")

# 3. Log in to Hugging Face
login(token="")
# 4. Create a new repo on Hugging Face
#repo = Repository(local_dir=repo_name, clone_from=f"PrashantBhalbar/{repo_name}")
#upload_file(path_or_fileobj="exam_score_predictor.pkl",
 #           path_in_repo="exam_score_predictor.pkl",
 #           repo_id="mujawarjaved/test",
 #           repo_type="model")
# 5. Move the model to the repo folder
#import shutil
#shutil.move("exam_score_predictor.pkl", repo_name)

# 6. Upload the model
#repo.push_to_hub(commit_message="Initial model upload")
model_filename = "exam_score_predictor.pkl"
repo_name = "exam-score-predictor_test_javed"  # Change as needed
create_repo(repo_name, private=False)  # Set private=True if you want a private repo
repo_id = "mujawarjaved/exam-score-predictor_test_javed"  # Example: "PrashantBhalbar/exam-score-predictor"
repo_type = "model"  # For model repositories


# Clone repo locally
repo = Repository(local_dir=repo_name, clone_from=f"mujawarjaved/{repo_name}")

# Move model file to repo
os.rename(model_filename, f"{repo_name}/{model_filename}")

# Push to Hugging Face
repo.push_to_hub(commit_message="Initial model upload")