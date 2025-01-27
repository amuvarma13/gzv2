mdn = "amuvarma/3b-zuckreg-convo"
from vllm import ModelRegistry, LLM, SamplingParams
import inspect

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model=mdn)

print(llm)

# Inspect the generate function
print(inspect.getsource(llm._run_engine))

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
