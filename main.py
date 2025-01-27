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

def generate_output(prompt, llm, sampling_params):
    start_time = time.time()
    
    # Get input token count
    input_tokens = len(tokenizer.encode(prompt))
    
    # Generate output
    output = llm.generate([prompt], sampling_params)[0]
    generated_text = output.outputs[0].text
    
    # Get output token count and calculate time
    output_tokens = len(tokenizer.encode(generated_text))
    total_tokens = input_tokens + output_tokens
    elapsed_time = time.time() - start_time
    tokens_per_second = total_tokens / elapsed_time
    
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    
    return generated_text

# prompt = "Here is a short story about a dragon:"
# sampling_params = SamplingParams(max_tokens=500)
# generate_output(prompt, llm, sampling_params)

