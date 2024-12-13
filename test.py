from transformers import AutoModelForCausalLM, AutoTokenizer

mdn = AutoModelForCausalLM.from_pretrained("models/checkpoint-14374")

print(mdn)