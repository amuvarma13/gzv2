from transformers import AutoModelForCausalLM, AutoTokenizer

mdn = AutoModelForCausalLM.from_pretrained("./mymodel")

print(mdn)