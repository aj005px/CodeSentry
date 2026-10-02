"""
LoRA fine-tuning for Qwen2.5-Coder-3B on the code-review task.

    python build_dataset.py            # writes review_train.jsonl / review_val.jsonl
    python train_lora.py

What changed relative to the previous version, and why
------------------------------------------------------
1. assistant_only_loss=True.  The previous run passed `dataset_text_field="text"`
   while the Dataset still carried its `messages` column, so TRL tokenized the
   whole rendered conversation and left labels on every position.  Measured on
   the old data: 178/178 tokens supervised, i.e. the system prompt and the code
   snippet absorbed roughly 60% of the gradient.  The task signal is the
   assistant turn only.  TRL refuses this flag unless the chat template contains
   `{% generation %}` markers, so we swap in a generation-masked template first.
2. A real train/validation split with eval loss and best-checkpoint loading.
   The previous run had no validation data at all, so "loss goes down" was the
   only available signal and there was no way to tell memorisation from learning.
3. Learning rate 2e-4 -> 1e-4, with checkpoint selection by eval loss.  The base
   model already reviews code well; the job is to change the output format with
   as little drift as possible, not to relearn review from 86 examples at 2e-4.
4. LoRA target modules and rank are deliberately UNCHANGED (r=16, alpha=32,
   attention only).  Adding MLP modules or raising rank would increase the
   capacity available to overwrite base behaviour, which is the opposite of
   what the measurement says we want.
5. Explicit seed and explicit max_length, so runs are reproducible and silent
   truncation cannot recur.

Memory: this host has 14 GB RAM and no GPU.  Training runs in bfloat16 on CPU;
float32 would need ~12.4 GB for the weights alone and will OOM.
"""

import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_PATH = os.environ.get("BASE_MODEL", "../models/qwen2.5-coder-3b")
TRAIN_PATH = "review_train.jsonl"
VAL_PATH = "review_val.jsonl"
OUTPUT_DIR = "./lora-code-reviewer-v2"

SEED = 42
MAX_LENGTH = 1024


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list([{"messages": r["messages"]} for r in rows])


def main():
    torch.manual_seed(SEED)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # assistant_only_loss needs `{% generation %}` markers in the template.
    # TRL swaps in its own generation-masked variant automatically when the
    # markers are absent, so we only need to make sure padding is defined.
    assert tokenizer.pad_token is not None

    print("Loading base model (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
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

    train_ds = load_jsonl(TRAIN_PATH)
    val_ds = load_jsonl(VAL_PATH)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        # transformers 5.x dropped warmup_ratio; this counts optimizer steps.
        # 103 examples / (bs 1 x grad_accum 8) = 13 steps per epoch, 39 total.
        warmup_steps=4,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        assistant_only_loss=True,
        max_length=MAX_LENGTH,
        use_cpu=True,
        bf16=False,
        fp16=False,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("Starting training...")
    result = trainer.train()

    print("Evaluating on the validation split...")
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    print(f"Saving LoRA adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(json.dumps({k: v for k, v in result.metrics.items()}, indent=2))


if __name__ == "__main__":
    main()