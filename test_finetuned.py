"""
Loads the base Qwen2.5-Coder-3B model + your trained LoRA adapter together,
so you can test the fine-tuned model and compare it against your Phase 2
baseline results.

Requires: pip install peft (in addition to what test.py already needed)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

BASE_MODEL_PATH = "./models/qwen2.5-coder-3b"
ADAPTER_PATH = "./dataset_project/lora-code-reviewer"

# Same import as test.py: inference must use the exact prompt used in training.
from dataset_project.build_dataset import SYSTEM_PROMPT

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype="bfloat16",
    low_cpu_mem_usage=True,
)
base_model = base_model.to("cpu")

print("Loading LoRA adapter on top of base model...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.to("cpu")
model.eval()

gen_config = GenerationConfig(
    max_new_tokens=400,
    do_sample=True,
    temperature=0.2,
)


def review(code: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Review this code:\n\n```\n{code}\n```"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    outputs = model.generate(**inputs, generation_config=gen_config)
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


print("\nFine-tuned model loaded! Paste code, end with a line containing only 'END'.\n" + "-" * 40)

while True:
    print("\nPaste code:")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    code = "\n".join(lines)

    if not code.strip():
        continue

    print("\n--- Fine-tuned Review ---")
    print(review(code))
    print("-" * 40)
