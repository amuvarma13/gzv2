import torch
import transformers
from transformers import Trainer, TrainingArguments
import torchaudio

from gzf import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)



from transformers import Wav2Vec2Config, LlamaConfig
import torch
import transformers
from transformers import Trainer, TrainingArguments
import torchaudio


from gzf import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)



import torch
device = "cpu"
dtype= torch.float32
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
    print(f"Using {device} device")
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
    print(f"Using {device} device")



model_id = "amuvarma/convo-tts-tune-7contentonly"


config = GazelleConfig(
    audio_model_id="facebook/wav2vec2-base-960h", 
    text_model_id=model_id, 
    audio_token_index = 134411, 
    vocab_size=134411,

    )

model = GazelleForConditionalGeneration(config).to(dtype=dtype)

tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})
# Don't forget to resize model embeddings if you have a model:
print("model device", model.device)
model.resize_token_embeddings(len(tokenizer))

print(model)

import wandb
wandb.init(
    project="colab-a100-40gb",
    name = "vmllama3bchat-500k-8h100s-2"
    )

file_path = 'transcribe_exps.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        expressions = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"The file {file_path} does not exist.")
except IOError:
    print(f"An error occurred while reading the file {file_path}.")


from datasets import load_dataset
dsn = "amuvarma/mls-eng-10k-dev-3k"
ds = load_dataset(dsn)




import torch
model = model.to(dtype=dtype)
# First freeze all parameters
for param in model.parameters():
   param.requires_grad = False

# Then unfreeze just the multi_modal_projector
# First set requires_grad
for name, param in model.named_parameters():
    if "multi_modal_projector" in name:
        param.requires_grad = True
        torch.nn.init.normal_(param, mean=0.0, std=0.02)

# Print to verify
for name, param in model.named_parameters():
   if param.requires_grad:
       print(f"Trainable: {name} - {param.shape}")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

print(f"\nTrainable parameters: {trainable_params:,}")
print(f"All parameters: {all_params:,}")

audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

from datasets import Dataset

dataset = ds["train"]


def inference_collator(audio_input, ass_res, instruction="Transcribe the following \n<|audio|>"):
    audio_values = audio_processor(
        audio=audio_input, return_tensors="pt", sampling_rate=16000
    ).input_values

    user_phrase = "<|audio|>"
    user_tokens = tokenizer.encode(user_phrase)
    assistant_tokens = tokenizer.encode(ass_res)

    print("user_tokens", user_tokens)

    user_tokens = tokenizer.encode(user_phrase)

    # Adding the specified tokens at start and end
    start_tokens = [128259, 128000]
    end_tokens = [128009, 128260, 128261]

    # Combine all tokens while keeping batch dimension
    user_tokens = torch.tensor([start_tokens + user_tokens + end_tokens ])
    labels = torch.tensor([start_tokens + user_tokens + end_tokens + assistant_tokens + [128009]])
    true_labels = torch.full_like(labels, -100)
    true_labels[:, user_tokens.shape[1]:] = labels[:, user_tokens.shape[1]:]

    print("true_labels", true_labels)
    print("input_ids", labels)

    attention_mask = torch.ones_like(labels)


    return {
        "audio_values": audio_input.to(model.device).to(model.dtype),
        "input_ids": labels.to(model.device),
        "labels": true_labels.to(model.device),
        "attention_mask": attention_mask.to(model.device)
    }


import random
class AudioChatDataCollator:
    def __init__(self, instruction="Transcribe the following \n<|audio|>"):
        self.instruction = instruction

    def __call__(self, features):
        audio = torch.tensor([features[0]["audio"]["array"]])
        assistant_response = features[0]["transcript"]

        random_expression = random.choice(expressions)



        batch = inference_collator(audio, assistant_response, random_expression)

        return {
            "audio_values": batch["audio_values"].cpu(),
            "input_ids": batch["input_ids"].cpu(),
            "labels": batch["labels"].cpu(),
            "attention_mask": batch["attention_mask"].cpu()
        }
    
training_args = TrainingArguments(
    output_dir="./audio-chat-test",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  # Changed to 16
    num_train_epochs=1,
    learning_rate=2e-3,  # Changed to 2*10^-3
    save_strategy="no",
    logging_steps=1,
    evaluation_strategy="no",
    report_to="wandb",
    push_to_hub=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=AudioChatDataCollator(),
)

trainer.train()