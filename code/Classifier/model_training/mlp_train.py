import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from datasets import load_from_disk
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

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


# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, label_mapping):
        self.texts = texts
        self.labels = [label_mapping[label] for label in labels]  # 将标签字符串转换为整数标签
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# MLP模型定义
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


dataset_dir = "/huggingface/mapneo_processed_data"
output_dir = "/huggingface/MLP0831"
PRETRAINED_MODEL = "/huggingface/models/Meta-Llama-3-8B-Instruct"
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = os.path.join(output_dir, "mlp_classifier.pth")
LOG_FILE_PATH = os.path.join(output_dir, "training_log.txt")
RESULT_FILE_PATH = os.path.join(output_dir, "training_result.txt")
ACC_FILE_PATH = os.path.join(output_dir, "acc.txt")
os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
tokenizer.pad_token = tokenizer.eos_token

train_data = load_from_disk(os.path.join(dataset_dir, "train_data"))
test_data = load_from_disk(os.path.join(dataset_dir, "test_data"))

train_texts = train_data['text']
train_labels = train_data['label']
test_texts = test_data['text']
test_labels = test_data['label']

train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH, label_mapping)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LENGTH, label_mapping)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = MLPClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_accuracy = 0.0
best_epoch = 0
with open(LOG_FILE_PATH, "w") as log_file, open(RESULT_FILE_PATH, "w") as result_file, open(ACC_FILE_PATH,"w") as acc_file:
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        log_file.write(f"\n==============================================\n")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            log_file.write(f"TRAINING - Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}\n")
            progress_bar.set_postfix(batch_loss=loss.item())
        log_file.write(f"\n==============================================\n")
        model.eval()
        total_correct = 0
        total_samples = 0
        total_eval_loss = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                total_eval_loss += loss.item()
                result_file.write(
                    f"TESTING - Epoch {epoch + 1}, Batch {batch_idx + 1}\n"
                    f"\noutput={outputs}\n"
                    f"\nlabels={labels}\n")
                predictions = torch.argmax(outputs, dim=1)
                correct = (predictions == labels).sum().item()
                accuracy = correct / labels.size(0)
                log_file.write(
                    f"TESTING - Epoch {epoch + 1}, Batch {batch_idx + 1}, Accuracy: {accuracy:.4f}, Loss: {loss.item():.4f}\n")
                total_correct += correct
                total_samples += labels.size(0)

        epoch_accuracy = total_correct / total_samples
        epoch_eval_loss = total_eval_loss / len(test_loader)

        acc_file.write(
            f"Epoch {epoch + 1}/{EPOCHS}, Epoch Average Accuracy: {epoch_accuracy:.4f}, Epoch Average Loss: {epoch_eval_loss:.4f}\n")

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model saved at epoch {best_epoch} with accuracy {best_accuracy:.4f}")

    print(f"Training completed. Best model saved from epoch {best_epoch} with accuracy {best_accuracy:.4f}.")