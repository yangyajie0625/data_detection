from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/huggingface/models/Meta-Llama-3-8B",trust_remote_code=True)
bos_token = tokenizer.bos_token
sampling_params = SamplingParams(temperature=1.0,max_tokens=512)

llm = LLM(model="/huggingface/models/Meta-Llama-3-8B",
          trust_remote_code=True,
          tensor_parallel_size=8)
bos_token_list = [bos_token] * 100000

output_file = "llama3_outputs_100k.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    outputs = llm.generate(bos_token_list, sampling_params=sampling_params)
    for output in tqdm(outputs, desc="Saving sequences"):
        generated_text = output.outputs[0].text
        json_record = json.dumps({"text": generated_text}, ensure_ascii=False)
        f.write(json_record + "\n")