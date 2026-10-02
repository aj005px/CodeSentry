"""
LoRA fine-tuning script for Qwen2.5-Coder-3B on your code-review dataset.

Run this from inside dataset_project/, with training_data.jsonl already built
(via build_dataset.py).

Install requirements first (run in your .venv):
    pip install peft trl transformers accelerate datasets torch

This is a small-scale first run intended to prove the pipeline works end to
end, not to produce a great model -- you only have ~30 examples right now.
Expect it to run fast and not necessarily produce dramatically better output;
the goal here is confirming the mechanics (loading, LoRA config, training loop,
saving) all work before you invest in growing the dataset further.
"""

import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_PATH = "../models/qwen2.5-coder-3b"  # adjust if your models folder is elsewhere
DATA_PATH = "training_data.jsonl"
OUTPUT_DIR = "./lora-code-reviewer"


def load_training_data(path: str) -> Dataset:
    """Load our JSONL file and reformat into plain text using the chat template."""
    examples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            examples.append({"messages": row["messages"]})
    return Dataset.from_list(examples)


def main():
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype="bfloat16",
        low_cpu_mem_usage=True,
    )

    print("Applying LoRA config...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    dataset = load_training_data(DATA_PATH)
    print(f"Loaded {len(dataset)} training examples.")

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_example)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,  # small dataset -- a few passes is reasonable
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        use_cpu=True,
        bf16=False,
        fp16=False,
        report_to="none",
        dataset_text_field="text",
    )

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done. Load it later with peft's PeftModel.from_pretrained().")


if __name__ == "__main__":
    main()
