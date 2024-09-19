## data_detection

### 1. Classifier

#### data_process

**data_process.py**  
Processes the Matrix dataset by randomly selecting a certain number of data entries from each category, saving the valid text to the `text` field of a jsonl file, and the category to the `label` field.

#### model_training

**mlp_train.py**  
Trains the MLP classifier and saves the model from the epoch with the highest accuracy to `MODEL_SAVE_PATH`.

**llama3_train.py**  
Fine-tunes the Llama3 model for classification tasks.

**merge.py**  
Loads the LoRA fine-tuned model and merges it with the Llama3 model, then saves the result.

**llama3_eval.py**  
Selects a certain amount of data from the dataset to evaluate the classification performance of the Llama3 model.

#### inference_and_classify

**inference_from_bos.py**  
Generates text from the model starting with the `bos_token` and saves it to `output_file`.

**mlp_classification.py**  
Loads the tokenizer from `PRETRAINED_MODEL` and the MLP from `MODEL_SAVE_PATH`, classifies the data from `input_jsonl_path`, and saves the result to `output_jsonl_path`.

**llama3_classification.py**  
Loads the Llama3 model from `model_path`, classifies the data from `input_jsonl_path`, and saves the result to `output_jsonl_path`.

#### result_analyze

**counter.py**  
Counts the number and proportion of each category in the classification result files of both MLP and Llama3.

**find_discrepancies.py**  
Identifies the data where the `label` classifications of MLP and Llama3 are inconsistent and saves them to `output_file`.

### 2. Perplexity

**sample.py**  
Randomly selects a total of `target_sample_size` entries from each jsonl file in the Matrix dataset, extracts the valid fields, and saves them.

**calculate_perplexity.py**  
Calculates the `perplexity` and `loss` for each file and computes the average `perplexity` and `loss` across all files, saving the results to `output_file`.
