import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Llama Model
model_name = "meta-llama/Meta-Llama-3-8B"  # You can also use a smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load user history data
df = pd.read_csv("user_history.csv")

# Convert data into training format
def format_input(row):
    return f"User visited: {', '.join(eval(row['visited_pages']))}. Recommend next page: "

df['input_text'] = df.apply(format_input, axis=1)
df['output_text'] = df['visited_pages'].apply(lambda x: eval(x)[-1])  # Last visited page as prediction

# Prepare tokenized data
def tokenize_function(example):
    return tokenizer(example['input_text'], text_target=example['output_text'], padding="max_length", truncation=True)

train_data = [tokenize_function(row) for _, row in df.iterrows()]

# Fine-tune Llama Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):  # Train for 3 epochs
    for data in train_data:
        input_ids = torch.tensor(data['input_ids']).unsqueeze(0).to(device)
        labels = torch.tensor(data['labels']).unsqueeze(0).to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

# Save trained model
model.save_pretrained("./recommendation_model")
