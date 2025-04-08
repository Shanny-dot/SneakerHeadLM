from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, TrainingArguments, Trainer
import torch
import json

# Load and flatten the SQuAD-style dataset
with open("qa_model/squad_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Transform into flat list
flat_data = []
for article in raw_data["data"]:
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            answer = qa["answers"][0]["text"]
            answer_start = qa["answers"][0]["answer_start"]

            flat_data.append({
                "id": qa["id"],
                "context": context,
                "question": question,
                "answer": answer,
                "answer_start": answer_start
            })

# Save flattened version temporarily for HuggingFace dataset loader
with open("qa_model/flat_data.json", "w", encoding="utf-8") as f:
    json.dump(flat_data, f)

# Load with HF Datasets
dataset = load_dataset("json", data_files={"train": "qa_model/flat_data.json"})["train"]

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Preprocess function
def preprocess(example):
    inputs = tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offset = inputs.pop("offset_mapping")[0]
    start_char = example["answer_start"]
    end_char = start_char + len(example["answer"])

    sequence_ids = inputs.sequence_ids(0)
    # Find start and end token indices
    token_start_index = sequence_ids.index(1)
    token_end_index = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

    start_token = end_token = None
    for idx in range(token_start_index, token_end_index + 1):
        if offset[idx][0] <= start_char and offset[idx][1] > start_char:
            start_token = idx
        if offset[idx][0] < end_char and offset[idx][1] >= end_char:
            end_token = idx
            break

    if start_token is None or end_token is None:
        start_token = end_token = 0  # fallback if mapping fails

    inputs["start_positions"] = torch.tensor(start_token)
    inputs["end_positions"] = torch.tensor(end_token)
    return {k: v.squeeze() for k, v in inputs.items()}

# Apply preprocessing
encoded_dataset = dataset.map(preprocess)

# Load model
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Training args (fixed for 2025 Transformers version)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",  # Changed from evaluation_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Might tweak to 4 for GTX 1650
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=50,
    fp16=True,  # Added for GTX 1650 optimization
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Save model
model.save_pretrained("qa_model/sneaker_qa_model")
tokenizer.save_pretrained("qa_model/sneaker_qa_model")