from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_PATH = "./models/qwen2.5-coder-3b"

# Winning prompt from testing: full schema, no "if no issues" escape-hatch line
# (that line was suppressing real bug detection in smaller-model testing).
SYSTEM_PROMPT = """You are a code review assistant. Given a code snippet, \
identify issues and return them in this exact format for each issue found:

Category: [Bug|Performance|Quality|Security]
Severity: [Low|Medium|High]
Line: <line number>
Problem: <one-line description>
Explanation: <why this is a problem>
Suggested Fix: <concrete fix>
"""

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="bfloat16",
    low_cpu_mem_usage=True,
)
model = model.to("cpu")

gen_config = GenerationConfig(
    max_new_tokens=400,
    do_sample=True,
    temperature=0.2,  # lower for consistency now that prompt is validated
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


print("\nModel loaded! Paste code, end with a line containing only 'END'.\n" + "-" * 40)

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

    print("\n--- Review ---")
    print(review(code))
    print("-" * 40)
