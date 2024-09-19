from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

base_model="/huggingface/models/Meta-Llama-3-8B-Instruct"
finetuned_model="/huggingface/fine-tuned-model"
tokenizer = AutoTokenizer.from_pretrained(base_model)

base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model_reload, finetuned_model)
model = model.merge_and_unload()
model_dir = "/huggingface/merged_llama3"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)