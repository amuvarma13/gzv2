import time
from transformers import AutoTokenizer
from mm_model_vllm import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
)

from transformers import AutoConfig, AutoModel
AutoConfig.register("orpheus", OrpheusConfig)
AutoModel.register(OrpheusConfig, OrpheusForConditionalGeneration)

from vllm import ModelRegistry, LLM, SamplingParams
ModelRegistry.register_model("OrpheusForConditionalGeneration", OrpheusForConditionalGeneration)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="./orpheus")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")