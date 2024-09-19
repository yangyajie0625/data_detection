import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)

import bitsandbytes as bnb
import json

system_prompt="You are a helpful AI assistant for machine learning and data processing"
user_prompt="Classify the following text into one of these nine categories: common-crawl, code, book, paper, instruction, exam, news, wiki, patent. Output only the category name without any additional comments. Here is a text that needs to be classified:"

df = pd.read_json("/huggingface/Matrix/merged_data.jsonl", lines=True)
X_test = df.sample(n=1000, random_state=66).reset_index(drop=True)

model_path = "/huggingface/merged_llama3"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
        model_path,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer.pad_token_id = tokenizer.eos_token_id
max_length = 1024


def truncate_text(text, tokenizer, max_length):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def generate_test_prompt(data_point, tokenizer, max_length):

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}{truncate_text(data_point["text"], tokenizer, max_length)} You should output:<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n """.strip()
    return prompt


# X_train.loc[:,'text'] = X_train.apply(lambda row: generate_prompt(row, tokenizer, max_length), axis=1)
y_true = X_test.loc[:, 'label']
X_test = pd.DataFrame(X_test.apply(lambda row: generate_test_prompt(row, tokenizer, max_length), axis=1),
                      columns=["text"])


# print(X_train.label.value_counts())
# train_data = Dataset.from_pandas(X_train[["text"]])

def predict(test, model, tokenizer, output_file="result0831.json"):
    y_pred = []
    predictions = []
    categories = ["common-crawl", "code", "book", "paper", "instruction", "exam", "news", "wiki", "patent"]
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=5,
                        temperature=0.1)

        result = pipe(prompt)
        answer = result[0]['generated_text'].split("assistant")[-1].strip()

        for category in categories:
            if category.lower() in answer.lower():
                y_pred.append(category)
                break
        else:
            y_pred.append("none")

        predictions.append({
            "text": prompt,
            "response": result[0]['generated_text']
        })

    # 将predictions列表保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    return y_pred


def evaluate(y_true, y_pred, output_file="eval083.txt"):
    labels = ["common-crawl", "code", "book", "paper", "instruction", "exam", "news", "wiki", "patent"]
    mapping = {label: idx for idx, label in enumerate(labels)}

    def map_func(x):
        return mapping.get(x, -1)

    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'Accuracy: {accuracy:.3f}\n')

        unique_labels = set(y_true_mapped)

        for label in unique_labels:
            label_indices = [i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label]
            label_y_true = [y_true_mapped[i] for i in label_indices]
            label_y_pred = [y_pred_mapped[i] for i in label_indices]
            label_accuracy = accuracy_score(label_y_true, label_y_pred)
            f.write(f'Accuracy for label {labels[label]}: {label_accuracy:.3f}\n')

        class_report = classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped, target_names=labels,
                                             labels=list(range(len(labels))))
        f.write('\nClassification Report:\n')
        f.write(class_report)

        conf_matrix = confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped, labels=list(range(len(labels))))
        f.write('\nConfusion Matrix:\n')
        f.write(str(conf_matrix))



y_pred = predict(X_test, model, tokenizer)
evaluate(y_true, y_pred)