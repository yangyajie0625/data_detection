import json
import random
from tqdm import tqdm

# 定义每个数据集的类别及其对应的文件名和抽取数量
categories = {
    "common-crawl": {
        "cc_en.0000.jsonl": 5000,
        "cc_zh.0000.jsonl": 5000
    },
    "code": {
        "code_code.0000.jsonl": 10000
    },
    "book": {
        "book_all.0000.jsonl": 10000
    },
    "paper": {
        "paper_all.0000.jsonl": 10000
    },
    "instruction": {
        "instruction_all.0000.jsonl": 10000
    },
    "exam": {
        "exam_biology.0000.jsonl": 4500,
        "exam_math.0000.jsonl": 4500,
        "exam_politics.0000.jsonl": 1000,
    },
    "news": {
        "news_finance.0000.jsonl": 5000,
        "news_politics.0000.jsonl": 5000
    },
    "wiki": {
        "wiki_all.0000.jsonl": 10000
    },
    "patent": {
        "patent_all.0000.jsonl": 10000
    }
}

output_file = "mapneo_data.jsonl"
with open(output_file, 'w', encoding='utf-8') as outfile:
    for label, files in categories.items():
        for file_name, sample_size in files.items():
            with open(file_name, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()
                if sample_size > len(lines):
                    sample_size = len(lines)
                sampled_lines = random.sample(lines, sample_size)
                for line in tqdm(sampled_lines, desc=f"Processing {file_name}", unit="lines"):
                    data = json.loads(line)
                    # 提取text字段，如果不存在则提取content字段
                    text = data.get("text", data.get("content", ""))
                    new_data = {"text": text, "label": label}
                    outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")
print("数据处理完成。")


