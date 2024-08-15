import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

# 设置环境变量以使用特定的GPU，并调整max_split_size_mb
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# 加载数据集
dataset = load_dataset('json', data_files='/data/coding/RLTF/data/data-oss_instruct-decontaminated.jsonl')

# 将数据集划分为训练集和验证集
dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)  # 10%作为验证集

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained('/data/coding/model/deepseek/deepseek-coder-6___7b-base')
max_length = 512

# 定义数据处理函数
def preprocess_function(examples):
    inputs = [f"Problem: {problem}\nSolution:" for problem in examples['problem']]
    targets = examples['solution']
    
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, truncation=True, padding='max_length')
    
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_ids]
        for label_ids in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 处理数据集
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 定义数据整理函数
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# DeepSpeed配置
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9
    },
    "gradient_clipping": 1.0,
    "fp16": {
        "enabled": True
    }
}

# 定义训练参数
training_args = TrainingArguments(
    output_dir='/data/coding/RLTF/data/new_result',
    evaluation_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,
    logging_steps=500,
    gradient_accumulation_steps=8,
    deepspeed=deepspeed_config
)

# 初始化模型
model = AutoModelForCausalLM.from_pretrained('/data/coding/model/deepseek/deepseek-coder-6___7b-base')

# 启用梯度检查点
model.gradient_checkpointing_enable()
model.config.use_cache = False
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 释放未使用的缓存
torch.cuda.empty_cache()

# 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
)

trainer.train()
