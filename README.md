## data_detection

### 1. Classifier

#### data_process

**data_process.py**  
处理Matrix数据集，从每个类别的数据中随机抽取一定数量，将其中有效文本保存到jsonl文件的`text`字段，类别保存到`label`字段。

#### model_training

**mlp_train.py**  
训练MLP分类器，保存准确率最高的epoch的模型到`MODEL_SAVE_PATH`。

**llama3_train.py**  
微调Llama3模型用于分类任务。

**merge.py**  
加载LoRA微调后的模型与Llama3模型合并后保存。

**llama3_eval.py**  
从数据集中抽取一定数量的数据，评估Llama3的分类效果。

#### inference_and_classify

**inference_from_bos.py**  
让模型从`bos_token`开始生成文本，并保存至`output_file`。

**mlp_classification.py**  
加载路径为`PRETRAINED_MODEL`的tokenizer和路径为`MODEL_SAVE_PATH`的MLP，对`input_jsonl_path`的数据进行分类，结果保存至`output_jsonl_path`。

**llama3_classification.py**  
加载路径为`model_path`的Llama3模型，对`input_jsonl_path`的数据进行分类，结果保存至`output_jsonl_path`。

#### result_analyze

**counter.py**  
统计MLP和Llama3分类结果文件中各个类别的数量及所占比例。

**find_discrepancies.py**  
找出MLP和Llama3分类`label`不一致的数据并保存至`output_file`。

### 2. Perplexity

**sample.py**  
从Matrix数据集的每个jsonl文件中随机抽取总共`target_sample_size`条数据，提取其中有效字段并保存。

**calculate_perplexity.py**  
计算每个文件的`perplexity`和`loss`，并计算所有文件的平均`perplexity`和`loss`，结果保存至`output_file`。
