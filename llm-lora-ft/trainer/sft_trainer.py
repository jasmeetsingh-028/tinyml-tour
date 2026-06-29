import torch
import tomllib
from datasets import load_dataset


from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.load_medical_o1_dataset import get_medical_resoning_sft
from trainer.custom_checkpoint import SaveEveryNEpochsCallback


def get_trainer(model, dataset, custom_epoch_callback):


    with open("configs/configs.toml", "rb") as f:
        configs = tomllib.load(f)

    # ----- LoRA config -----
    lora_config = LoraConfig(**configs["lora"])

    # ----- peft model -----
    model = get_peft_model(model, lora_config)

    # ----- SFT config -----
    sft_config =  SFTConfig(
        **configs["sft"],
        bf16=torch.cuda.is_available() 
    )

    # ----- Trainer -----
    trainer = SFTTrainer(
        model = model,
        args = sft_config,
        train_dataset = dataset,
         callbacks=[custom_epoch_callback]
    )

    return trainer


if __name__ == "__main__":
    
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # ~1GB
    DATASET_NAME = "FreedomIntelligence/medical-o1-reasoning-SFT"

    dataset = load_dataset(DATASET_NAME, "en")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    custom_callback = SaveEveryNEpochsCallback( 
        n=3,
        output_dir="./lora_out",
        tokenizer=tokenizer
        )

    train_ds, eval_ds = get_medical_resoning_sft(dataset, tokenizer)
    trainer = get_trainer(model, train_ds, custom_callback)


