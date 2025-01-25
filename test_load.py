from mm_model import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
)
from transformers import AutoModel, AutoTokenizer
import soundfile as sf

from pydub import AudioSegment
from snac import SNAC
import torch

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to("cuda")
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# 4. Add custom tokens
number_add_tokens = 7 * 4096 + 10
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]  # 6155 tokens
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})

# 2. Device and dtype setup
import torch
device = "cpu"
dtype = torch.float32
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
    print(f"Using {device} device")
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
    print(f"Using {device} device")

import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# 4. Add custom tokens
number_add_tokens = 7 * 4096 + 10
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]  # 6155 tokens
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})

mdn = "./orpheus"
m = AutoModel.from_pretrained(mdn)
print(m)