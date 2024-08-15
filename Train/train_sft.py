import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
import wandb


model = AutoModelForCausalLM.from_pretrained(model_name)  # 将模型移到GPU上
model.gradient_checkpointing_enable()
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True


run = wandb.init(
    project="TRAIN1",
    job_type="training",
)
dataset = load_dataset("json", data_files=dataset_name, split="train")

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
    gradient_accumulation_steps=4,
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

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    dataset_text_field="text",
    packing=False,
)

trainer.train()
