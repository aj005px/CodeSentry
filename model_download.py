from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
SAVE_PATH = "./models/qwen2.5-coder-3b"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

print("Saving...")
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"Model saved to {SAVE_PATH}")
