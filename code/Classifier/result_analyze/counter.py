import json
from collections import Counter
from tqdm import tqdm

def count_labels(jsonl_path):
    label_counter = Counter()
    total_count = 0

    with open(jsonl_path, mode='r', encoding='utf-8') as file:
        for line in tqdm(file, desc=f"Processing {jsonl_path}", unit=" lines"):
            data = json.loads(line)
            label = data.get('label')
            label_counter[label] += 1
            total_count += 1

    return label_counter, total_count


def print_label_counts(counter, total_count, file_name):
    print(f"\nLabel counts for {file_name}:")
    for label, count in counter.items():
        percentage = (count / total_count) * 100
        print(f"{label}: {count}, ({percentage:.2f}%)")


def main(file1, file2):
    label_counts_1, total_count_1 = count_labels(file1)
    print_label_counts(label_counts_1, total_count_1, file1)

    label_counts_2, total_count_2 = count_labels(file2)
    print_label_counts(label_counts_2, total_count_2, file2)


if __name__ == "__main__":
    jsonl_file_1 = "/huggingface/llama3_llama3_classified.jsonl"
    jsonl_file_2 = "/huggingface/yyj/llama3_mlp_classified_2.jsonl"

    main(jsonl_file_1, jsonl_file_2)
