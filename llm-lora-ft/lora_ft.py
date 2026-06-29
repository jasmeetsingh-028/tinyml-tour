import torch
from datasets import Dataset


from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from load_dataset import get_medical_resoning_sft


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # ~1GB
DATASET_NAME = "FreedomIntelligence/medical-o1-reasoning-SFT"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# get dataset

train_ds, eval_ds = get_medical_resoning_sft(DATASET_NAME, tokenizer)

print(train_ds[0]["text"])
print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")