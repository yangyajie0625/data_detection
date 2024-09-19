import torch
import torch.nn as nn
from transformers import AutoTokenizer
import jsonlines
from tqdm import tqdm
import os

label_mapping = {
    "common-crawl": 0,
    "code": 1,
    "book": 2,
    "paper": 3,
    "instruction": 4,
    "exam": 5,
    "news": 6,
    "wiki": 7,
    "patent": 8
}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

class MLPClassifier(nn.Module):
    def __init__(self, output_dim=9, vocab_size=129000, embedding_dim=256):
        super(MLPClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * MAX_LENGTH, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)  # [batch_size, max_length, embedding_dim]
        embedded = embedded.view(embedded.size(0), -1)  # 展平为 [batch_size, max_length * embedding_dim]
        x = self.relu(self.fc1(embedded))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

MAX_LENGTH = 512
MODEL_SAVE_PATH = "/huggingface/MLP0831/mlp_classifier.pth"
PRETRAINED_MODEL = "/huggingface/models/Meta-Llama-3-8B-Instruct"

input_jsonl_path = "/huggingface/llama3_outputs_100k.jsonl"
output_jsonl_path = "/huggingface/llama3_mlp_classified_2.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPClassifier()
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
tokenizer.pad_token = tokenizer.eos_token

def classify_text(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].squeeze().to(device)

    with torch.no_grad():
        output = model(input_ids.unsqueeze(0))  # 添加 batch 维度
        prediction = torch.argmax(output, dim=1).item()
        label = inverse_label_mapping[prediction]
    return label

# 读取 JSONL 文件并分类，使用 tqdm 显示进度条
with jsonlines.open(input_jsonl_path, mode='r') as reader, jsonlines.open(output_jsonl_path, mode='w') as writer:
    for obj in tqdm(reader, desc="Classifying texts", unit="text"):
        text = obj['text']
        predicted_label = classify_text(text)
        obj['label'] = predicted_label
        writer.write(obj)

print(f"Classification completed. Results saved to {output_jsonl_path}.")