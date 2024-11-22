import torch
import transformers
from transformers import Trainer, TrainingArguments
import torchaudio

from gazelle.gazelle.modeling_gazelle import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)

model_id = "meta-llama/Llama-2-7b-chat-hf"
nt = transformers.AutoTokenizer.from_pretrained(model_id)
token = nt.convert_ids_to_tokens(32000)
print(token)

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

from transformers import Wav2Vec2Config, LlamaConfig
import torch
import transformers
from transformers import Trainer, TrainingArguments
import torchaudio


from gazelle.gazelle import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)
# model_id = "meta-llama/Llama-2-7b-chat-hf"
audio_config = Wav2Vec2Config()

text_config = LlamaConfig()




config = GazelleConfig(audio_model_id="facebook/wav2vec2-base-960h", text_model_id=model_id)
model = GazelleForConditionalGeneration(config)
model = model.to(device, dtype=dtype)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})
# Don't forget to resize model embeddings if you have a model:
model.resize_token_embeddings(len(tokenizer))

import wandb
wandb.init(
    project="colab-a100-40gb",
    name = "llama3b-200k-6"
    )

file_path = 'transcribe_exps.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        expressions = [line.strip() for line in file if line.strip()]
    print(expressions)
except FileNotFoundError:
    print(f"The file {file_path} does not exist.")
except IOError:
    print(f"An error occurred while reading the file {file_path}.")


from datasets import load_dataset
dsn = "amuvarma/mls-eng-10k-200k"
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

    msgs = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": ass_res}
    ]

    labels = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True
    )

    user_msg = [{"role": "user", "content": instruction}, {"role":"assistant", "content":""}]
    user_tokens = tokenizer.apply_chat_template(
        user_msg, return_tensors="pt", add_generation_prompt=True
    )

    true_labels = torch.full_like(labels, -100)
    true_labels[:, user_tokens.shape[1]:] = labels[:, user_tokens.shape[1]:]

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
    gradient_accumulation_steps=16,  # Changed to 16
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
