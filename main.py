import time
from transformers import AutoTokenizer
from mm_model_vllm import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
)

from transformers import AutoConfig, AutoModel
AutoConfig.register("orpheus", OrpheusConfig)
AutoModel.register(OrpheusConfig, OrpheusForConditionalGeneration)
mdn = "./orpheus"

tokenizer = AutoTokenizer.from_pretrained(mdn)
model = AutoModel.from_pretrained(mdn)

print(model)

def generate_output(prompt):
    start_time = time.time()
    
    # Get input token count
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    print("inputs", inputs)
    
    # Generate output
    outs = model.generate(
        inputs,
        max_new_tokens=50,
        temperature=0.7,
        repetition_penalty=1.1,
        top_p=0.9,
        eos_token_id=128258,
    )
    
    total_tokens = outs
    elapsed_time = time.time() - start_time
    tokens_per_second = total_tokens / elapsed_time
    
    print(f"Prompt: {prompt!r}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    
    # return generated_text

prompt = "Here is a short story about a dragon:"
generate_output(prompt)
# sampling_params = SamplingParams(max_tokens=500)
# generate_output(prompt, llm, sampling_params)

