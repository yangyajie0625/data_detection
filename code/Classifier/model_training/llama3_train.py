import os
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_from_disk
import wandb

system_prompt="You are a helpful AI assistant for machine learning and data processing"
user_prompt="Classify the following text into one of these nine categories: common-crawl, code, book, paper, instruction, exam, news, wiki, patent. Output only the category name without any additional comments. Here is a text that needs to be classified:"
def process_func(example):
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}{example['text']} You should output:<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
    response = tokenizer(f"{example['label']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
run = wandb.init(
    project='2024-0817-2',
    name='train0817',
    job_type="training",
)
try:
    dataset_dir = "/huggingface/mapneo_processed_data"
    train_data = load_from_disk(os.path.join(dataset_dir, "train_data"))
    test_data = load_from_disk(os.path.join(dataset_dir, "test_data"))

    train_data = train_data.select(range(24000))
    test_data = test_data.select(range(6000))

    output_dir = "/huggingface/fine-tuned-model"
    model_path = "/huggingface/models/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()

    tokenized_train = train_data.map(process_func, remove_columns=train_data.column_names)
    tokenized_test = test_data.map(process_func, remove_columns=test_data.column_names)
    print(tokenizer.decode(tokenized_train[0]['input_ids']))
    print('='*10)
    print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_train[0]["labels"]))))

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        logging_steps=20,
        num_train_epochs=2,
        save_steps=300,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=300,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
finally:
    wandb.finish()