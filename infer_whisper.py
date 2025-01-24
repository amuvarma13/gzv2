from mm_model import (
    OrpheusConfig,
    OrpheusForConditionalGeneration,
)
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
  

text_model_id= "amuvarma/3b-zuckreg-convo"
mm_model_id = "amuvarma/3b-zuckreg-convo-projsnactune"

download_model(text_model_id)
download_model(mm_model_id)
model_id = text_model_id
config = OrpheusConfig(
    audio_model_id="facebook/wav2vec2-base-960h",
    text_model_id=model_id,
    audio_token_index=156939,
    vocab_size=156939,
)
model = OrpheusForConditionalGeneration(config).to(dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer))
special_config =  model.config

loaded_model_custom = OrpheusForConditionalGeneration.from_pretrained(mm_model_id, config=special_config, new_vocab_size=False)
loaded_model_custom = loaded_model_custom.to("cuda").to(torch.bfloat16)


import whisper

whisper_model = whisper.load_model("small")
import torchaudio
from IPython.display import Audio

def process_audio_tensor(audio, sample_rate=16000):
    audio = audio.to(torch.float32)
    duration_ms = (len(audio) / sample_rate) * 1000
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel, int(duration_ms / 20) + 1

sound = AudioSegment.from_mp3("3.mp3")
sound.export("recorded_audio.wav", format="wav")
test_audio1, sr1 = torchaudio.load("recorded_audio.wav")

def get_audio_features():
  test_audio, sr = torchaudio.load("recorded_audio.wav")
  dtype = torch.float16
  if sr != 16000:
      test_audio = torchaudio.transforms.Resample(sr, 16000)(test_audio)

  audio_input = test_audio.squeeze(0)
  mel, length = process_audio_tensor(audio_input)
  mel = mel.to(whisper_model.device)
  mel = mel.unsqueeze(0)
  audio_feature = whisper_model.embed_audio(mel)[0][:length]
  audio_feature = audio_feature.unsqueeze(0)
  return audio_feature

def redistribute_codes(code_list):
  layer_1 = []
  layer_2 = []
  layer_3 = []
  for i in range((len(code_list)+1)//7):
    layer_1.append(code_list[7*i])
    layer_2.append(code_list[7*i+1]-4096)
    layer_3.append(code_list[7*i+2]-(2*4096))
    layer_3.append(code_list[7*i+3]-(3*4096))
    layer_2.append(code_list[7*i+4]-(4*4096))
    layer_3.append(code_list[7*i+5]-(5*4096))
    layer_3.append(code_list[7*i+6]-(6*4096))

  codes = [torch.tensor(layer_1).unsqueeze(0).to("cuda"),
         torch.tensor(layer_2).unsqueeze(0).to("cuda"),
         torch.tensor(layer_3).unsqueeze(0).to("cuda")]
  audio_hat = snac_model.decode(codes)
  return audio_hat


def extract_tokens_after_value(tensor, target_start=128257, target_end=128258):
    tensor_list = tensor.tolist()

    start_index = tensor_list.index(target_start)
    try:
        end_index = tensor_list.index(target_end, start_index)
        return tensor_list[start_index + 1:end_index]
    except ValueError:
        return tensor_list[start_index + 1:]

def turn_outs_to_audio(outs):
  audio_list = extract_tokens_after_value(outs[0], 128257, 128258)
  token_to_find = 128257
  token_to_remove = 128263

  # Check if the token exists in the tensor
  token_indices = (outs == token_to_find).nonzero(as_tuple=True)

  if len(token_indices[1]) > 0:
      last_occurrence_idx = token_indices[1][-1].item()
      cropped_tensor = outs[:, last_occurrence_idx+1:]
  else:
      cropped_tensor = outs

  mask = cropped_tensor != token_to_remove
  cropped_tensor = cropped_tensor[mask].view(cropped_tensor.size(0), -1)

  processed_tensor = cropped_tensor - 128266
  original_shape = processed_tensor.shape
  new_dim_1 = (original_shape[1] // 7) * 7
  processed_tensor = processed_tensor[:, :new_dim_1]
  code_list = processed_tensor[0].tolist()
  samples = redistribute_codes(code_list)
  waveform = samples.detach().squeeze().to("cpu").numpy()
  sf.write("answer.wav", waveform, samplerate=24000)
  return Audio(samples.detach().squeeze().to("cpu").numpy(), rate=24000)


def get_all_embeds(existing_embeds, audio_features):
  print(type(audio_features))
  if type(audio_features) == type(""):
    text_tokens = tokenizer(text, return_tensors="pt").input_ids
    text_tokens = text_tokens.to(loaded_model_custom.device)
    text_embeds = loaded_model_custom.get_input_embeddings()(text_tokens)
    audio_embeds = text_embeds.to(dtype=torch.bfloat16).to(loaded_model_custom.device)
  else:

    audio_features = audio_features.to(dtype=torch.bfloat16).to(loaded_model_custom.device)
    audio_embeds = loaded_model_custom.multi_modal_projector(audio_features)


  start_token = torch.tensor([[128259, 128000]], dtype=torch.int64)
  end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)
  final_tokens = torch.tensor([[128262]], dtype=torch.int64)
  start_token = start_token.to(loaded_model_custom.device)
  end_tokens = end_tokens.to(loaded_model_custom.device)
  final_tokens = final_tokens.to(loaded_model_custom.device)
  start_embeds = loaded_model_custom.get_input_embeddings()(start_token)
  end_embeds = loaded_model_custom.get_input_embeddings()(end_tokens)
  final_embeds = loaded_model_custom.get_input_embeddings()(final_tokens)
  start_embeds = start_embeds.to(dtype=torch.bfloat16)
  end_embeds = end_embeds.to(dtype=torch.bfloat16)
  final_embeds = final_embeds.to(dtype=torch.bfloat16)
  if existing_embeds is not None:
    all_embeds = torch.cat([existing_embeds, start_embeds, audio_embeds, end_embeds], dim=1)
  else:
    all_embeds = torch.cat([start_embeds, audio_embeds, end_embeds], dim=1)
  print("all_embeds.shape", all_embeds.shape)
  inputs = {"inputs_embeds": all_embeds.to(loaded_model_custom.device).to(torch.bfloat16)}
  outs = loaded_model_custom.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        repetition_penalty=1.1,
        top_p=0.9,
        eos_token_id=128258,
  )
  output_embs = loaded_model_custom.get_input_embeddings()(outs)
  complete_embeds = torch.cat([all_embeds, output_embs, final_embeds], dim=1)
  return complete_embeds, outs


existing_embeds = None

audio_features = get_audio_features()
# existing_embeds = None
print(audio_features.shape)
# text = "Who are your guys's competitors?"
existing_embeds, outs = get_all_embeds(existing_embeds, audio_features)
print(tokenizer.decode(outs[0]))
turn_outs_to_audio(outs)