import random
import json
import os

output_dir = '../sampled_Matrix'
os.makedirs(output_dir, exist_ok=True)

def sample_data(file_path, sample_size):
    sampled_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) < sample_size:
            sampled_lines = lines
        else:
            sampled_lines = random.sample(lines, sample_size)

    extracted_texts = []
    for line in sampled_lines:
        data = json.loads(line.strip())
        text = data.get('text', '').strip()
        if not text:
            text = data.get('content', '').strip()
        extracted_texts.append({'text': text})
    return extracted_texts

categories = {
    "book": ["book_all.0000.jsonl"],
    "cc": ["cc_en.0000.jsonl", "cc_zh.0000.jsonl"],
    "code": ["code_code.0000.jsonl"],
    "exam": ["exam_biology.0000.jsonl", "exam_math.0000.jsonl", "exam_politics.0000.jsonl", "exam_QA.0000.jsonl"],
    "instruction": ["instruction_all.0000.jsonl"],
    "news": ["news_finance.0000.jsonl", "news_politics.0000.jsonl"],
    "paper": ["paper_all.0000.jsonl"],
    "patent": ["patent_all.0000.jsonl"],
    "wiki": ["wiki_all.0000.jsonl"]
}

target_sample_size = 10000
for category, files in categories.items():
    category_texts = []
    file_sample_sizes = [target_sample_size // len(files) for _ in files]
    for file, sample_size in zip(files, file_sample_sizes):
        sampled_texts = sample_data(file, sample_size)
        category_texts.extend(sampled_texts)


    category_output_file = os.path.join(output_dir, f'{category}_sampled.jsonl')
    with open(category_output_file, 'w', encoding='utf-8') as cat_file:
        for item in category_texts:
            json.dump(item, cat_file, ensure_ascii=False)
            cat_file.write('\n')

    print(f"{category}_sampled.jsonl is saved")

print("Sampled data saved to separate JSONL files for each category in ../sampled_Matrix.")
