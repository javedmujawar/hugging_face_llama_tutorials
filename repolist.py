from huggingface_hub import HfApi

api = HfApi()
user_repos = api.list_models("mujawarjaved")  # Replace with your username

# Print repo names
for repo in user_repos:
    print(repo.id)
