from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
import torch

# Load your fine-tuned model
model_path = "qa_model/sneaker_qa_model"
model = DistilBertForQuestionAnswering.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

model.eval()  # Important: Set model to evaluation mode

# 🧪 Sample QA loop
while True:
    context = input("\n📘 Enter the context (or type 'exit'): ").strip()
    if context.lower() == "exit":
        break

    question = input("❓ Enter your question: ").strip()

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index])
    )

    print(f"\n💡 Answer: {answer}")