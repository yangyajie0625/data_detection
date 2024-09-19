import json

def find_discrepancies(file1, file2, output_file):
    discrepancies = []

    with open(file1, mode='r', encoding='utf-8') as f1, open(file2, mode='r', encoding='utf-8') as f2:
        for line1, line2 in zip(f1, f2):
            data1 = json.loads(line1)
            data2 = json.loads(line2)

            text1 = data1.get('text')
            text2 = data2.get('text')
            label1 = data1.get('label')
            label2 = data2.get('label')

            if text1 == text2 and label1 != label2:
                discrepancies.append({
                    'text': text1,
                    'llama3_label': label1,
                    'mlp_label': label2
                })

    with open(output_file, mode='w', encoding='utf-8') as output:
        for item in discrepancies:
            output.write(json.dumps(item, ensure_ascii=False) + '\n')


llama3_file = "/huggingface/llama3_llama3_classified.jsonl"
mlp_file = "/huggingface/llama3_mlp_classified_2.jsonl"
output_file = "/huggingface/discrepancies.jsonl"

find_discrepancies(llama3_file, mlp_file, output_file)
