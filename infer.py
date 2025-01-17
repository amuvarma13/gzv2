from gzf import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)

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
 

from huggingface_hub import snapshot_download
def download_model(model_id):
  model_path = snapshot_download(
    repo_id=model_id,
    allow_patterns=[
        "config.json",
        "*.safetensors",
        "model.safetensors.index.json",
    ],
    ignore_patterns=[
        "optimizer.pt",
        "pytorch_model.bin",
        "training_args.bin",
        "scheduler.pt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.*"
    ]
)
  
audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

import torchaudio
from IPython.display import Audio

sound = AudioSegment.from_mp3("3.mp3")
sound.export("recorded_audio.wav", format="wav")
test_audio1, sr1 = torchaudio.load("recorded_audio.wav")
dtype = torch.float32

if sr1 != 16000:
    print("resampling audio")
    test_audio = torchaudio.transforms.Resample(sr1, 16000)(test_audio1)

print("test_audio",test_audio.shape)

audio_values1 = audio_processor(
    audio=test_audio, return_tensors="pt", sampling_rate=16000
).input_values


# audio_values = torch.cat(audio_values1, dim=0)

audio_values = audio_values1.squeeze(0)
print("audio_values",audio_values.shape)
#@title Create inference collator
def new_inference_collator():
    # user_phrase = "Okay so what would be a healthier breakfast option then? Can you tell me?"
    user_phrase = "<|audio|>"
    user_input_ids = tokenizer(user_phrase, return_tensors="pt").input_ids
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)


    user_tokens = torch.cat(
        [start_token, user_input_ids, end_tokens], dim=1)

    return user_tokens
user_tokens = new_inference_collator()



text_model_id = "amuvarma/snac-pretrain-2m-96000"
mm_model_id = "amuvarma/3b-2m-proj-checkpoint-14375-0"

# download_model(text_model_id)
# download_model(mm_model_id)
model_id = text_model_id
config = GazelleConfig(
    audio_model_id="facebook/wav2vec2-base-960h",
    text_model_id=model_id,
    audio_token_index=156939,
    vocab_size=156939,
)
model = GazelleForConditionalGeneration(config).to(dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer))
special_config =  model.config

loaded_model_custom = GazelleForConditionalGeneration.from_pretrained(mm_model_id, config=special_config, new_vocab_size=False)
loaded_model_custom = loaded_model_custom.to("cuda").to(torch.bfloat16)

myinputs= {
  "audio_values": audio_values.to(loaded_model_custom.device).to(torch.bfloat16),
  "input_ids": user_tokens.to(loaded_model_custom.device),
  # "input_ids": tokenizer("How does Facebook manage user privacy?", return_tensors="pt").input_ids.to("cuda")
}

# loaded_model_custom.eval()
import time
start_time = time.time()
with torch.no_grad():
  outs = loaded_model_custom.generate(
      **myinputs,
      max_new_tokens=50,
      temperature=0.3,
      repetition_penalty=1.1,
      top_p=0.9,
      eos_token_id=128258,
      )

end_time = time.time()
print(tokenizer.decode(outs[0]))