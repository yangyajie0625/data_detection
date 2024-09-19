import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os

model_path = '/huggingface/mapneo_7b'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,device_map="auto"
)

model.eval()


def calculate_average_loss(texts):
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            input_ids = tokenizer(text, return_tensors='pt', padding=False, truncation=True).input_ids.to('cuda')
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    return total_loss / total_tokens


def read_data(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            texts.append(data['text'])
    return texts


sampled_dir = '../sampled_Matrix'
categories = [
    "book_sampled.jsonl", "cc_sampled.jsonl", "code_sampled.jsonl",
    "exam_sampled.jsonl", "instruction_sampled.jsonl", "news_sampled.jsonl",
    "paper_sampled.jsonl", "patent_sampled.jsonl", "wiki_sampled.jsonl"
]

file_losses = {}
total_loss = 0.0
total_samples = 0

for category_file in tqdm(categories, desc="Calculating Losses"):
    file_path = os.path.join(sampled_dir, category_file)
    texts = read_data(file_path)
    average_loss = calculate_average_loss(texts)
    file_losses[category_file] = average_loss
    total_loss += average_loss * len(texts)
    total_samples += len(texts)

overall_average_loss = total_loss / total_samples
overall_perplexity = torch.exp(torch.tensor(overall_average_loss)).item()

output_file = 'perplexity_results.txt'
with open(output_file, 'w') as out_file:
    out_file.write("File Losses and Perplexities:\n")
    for file, loss in file_losses.items():
        perplexity = torch.exp(torch.tensor(loss)).item()
        out_file.write(f"{file}: Loss = {loss}, Perplexity = {perplexity}\n")

    out_file.write(
        f"\nOverall Average Loss = {overall_average_loss}, Overall Perplexity (mapneo) = {overall_perplexity}\n")

print(f"Perplexity results saved to {output_file}")
