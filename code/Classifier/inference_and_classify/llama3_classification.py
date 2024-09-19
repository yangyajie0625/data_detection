import json
from tqdm import tqdm
from vllm import LLM, SamplingParams

system_prompt = "You are a helpful AI assistant for machine learning and data processing."
user_prompt = "Classify the following text into one of these nine categories: common-crawl, code, book, paper, instruction, exam, news, wiki, patent. Output only the category name without any additional comments. Here is a text that needs to be classified:"

model_path = "/huggingface/merged_llama3"
input_jsonl_path = "/huggingface/llama3_outputs_100k.jsonl"
output_jsonl_path = "/huggingface/llama3_llama3_classified.jsonl"

llm = LLM(model=model_path,tensor_parallel_size=8)
sampling_params = SamplingParams(temperature=0.1)
def generate_test_prompt(text):
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}{text} You should output:<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n".strip()
    return prompt
def predict(input_jsonl_path, output_jsonl_path, llm):
    categories = ["common-crawl", "code", "book", "paper", "instruction", "exam", "news", "wiki", "patent"]
    prompts = []

    with open(input_jsonl_path, mode='r', encoding='utf-8') as reader:
        for line in tqdm(reader, desc="Preparing prompts", unit="text"):
            obj = json.loads(line)
            text = obj['text']
            prompt = generate_test_prompt(text)
            prompts.append((prompt, obj))

    results = llm.generate([p[0] for p in prompts], sampling_params)

    with open(output_jsonl_path, mode='w', encoding='utf-8') as writer:
        for result, (prompt, obj) in zip(results, prompts):
            answer = result.outputs[0].text.split("assistant")[-1].strip()

            for category in categories:
                if category.lower() in answer.lower():
                    obj['label'] = category
                    break
            else:
                obj['label'] = "none"

            writer.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print(f"Classification completed. Results saved to {output_jsonl_path}.")

predict(input_jsonl_path, output_jsonl_path, llm)