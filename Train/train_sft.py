import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
import wandb

# 设置环境变量


# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_name)  # 将模型移到GPU上
model.gradient_checkpointing_enable()
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

# 登录 wandb

run = wandb.init(
    project="TRAIN1",
    job_type="training",
)

# 加载数据集
dataset = load_dataset("json", data_files=dataset_name, split="train")

# 配置 LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
# Training arguments with DeepSpeed
training_arguments = TrainingArguments(
    output_dir="",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 增加梯度累积步数
    optim="paged_adamw_8bit",
    save_steps=25,
    logging_steps=30,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    gradient_checkpointing = True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.3,
    group_by_length=True,
    lr_scheduler_type="linear",
    report_to="wandb",
)

# 初始化 Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    dataset_text_field="text",
    packing=False,
)

# 训练模型
trainer.train()
