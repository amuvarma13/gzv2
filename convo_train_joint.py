import random
from datasets import load_dataset
import wandb
from datasets import Dataset

import torch
from tqdm import tqdm

from transformers import Wav2Vec2Config, LlamaConfig
import torch
import transformers
from transformers import Trainer, TrainingArguments
import torchaudio
from transformers import CONFIG_MAPPING

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import numpy as np



from gzf import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)

MODEL_FOR_CAUSAL_LM_MAPPING.register(
    "gazelle", GazelleForConditionalGeneration)
# CONFIG_MAPPING["gazelle"] = GazelleConfig


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


tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# 4. Add custom tokens
number_add_tokens = 7 * 4096 + 10 
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})


model_id = "./mymodel"
config = GazelleConfig(
    audio_model_id="facebook/wav2vec2-base-960h",
    text_model_id=model_id,
    audio_token_index=156939,
    vocab_size=156939,
)
model = GazelleForConditionalGeneration(config).to(dtype=dtype)
model.resize_token_embeddings(len(tokenizer))
special_config =  model.config

output_dir = "amuvarma/snac-e2e-projonly-3"


print("before loading")

model = GazelleForConditionalGeneration.from_pretrained(output_dir, config=special_config, new_vocab_size=True)

print("after loading")



dsn = "amuvarma/conversation-elias-5-0-t248-convo-both-full-snacced-ds"
# dsn = "amuvarma/mls-eng-10k-dev-3k"
ds = load_dataset(dsn, split="train")
# ds = ds.select(range(10000, 199999))
def remove_short_audio(dataset, min_seconds=1.0):
    indices_to_keep = []

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        audio = example['question_audio']
        duration = len(audio['array']) / audio['sampling_rate']
        if duration >= min_seconds:
            indices_to_keep.append(i)

    filtered_dataset = dataset.select(indices_to_keep)

    return filtered_dataset

# ds = remove_short_audio(ds)


for param in model.parameters():
    param.requires_grad = False

special_config = model.config
wandb.init(
    project="convo-tune",
    name="r0-23-12"
)

file_path = 'transcribe_exps.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        expressions = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"The file {file_path} does not exist.")
except IOError:
    print(f"An error occurred while reading the file {file_path}.")




dataset = ds


model = model.to(dtype=dtype)

print(model)
print(len(tokenizer))

# First freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Then unfreeze just the multi_modal_projector
# First set requires_grad
for name, param in model.named_parameters():
    if "multi_modal_projector" in name:
        param.requires_grad = True
    # if "language_model" in name:
    #     param.requires_grad = True
#         torch.nn.init.normal_(param, mean=0.0, std=0.02)

# Print to verify
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"Trainable: {name} - {param.shape}")

trainable_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

print(f"\nTrainable parameters: {trainable_params:,}")
print(f"All parameters: {all_params:,}")

audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)


print("creating collator")


def inference_collator(features):

    
    user_input_ids = user_input_ids = tokenizer("<|audio|>", return_tensors="pt").input_ids

    start_of_system = torch.tensor([[128256+8]], dtype=torch.int64)
    end_of_system = torch.tensor([[128256+9]], dtype=torch.int64)
    end_of_text = torch.tensor([[128009]], dtype=torch.int64)


    system_message = "You are an AI assistant who will answer the user's questions and follow the user's instructions."
    system_input_ids = tokenizer(system_message, return_tensors="pt").input_ids
    system_tokens = torch.cat(
        [start_of_system, system_input_ids, end_of_text, end_of_system],  dim=1)

    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)
    final_tokens = torch.tensor([[128009, 128257 ]], dtype=torch.int64)
    post_assistant_tokens = torch.tensor([[128258, 128262]])
    my_input_ids = system_tokens
    audio_inputs = []

    for i in range(6):
        if  f"user_{i}_text" in features[0] and features[0][f"user_{i}_text"] is not None:

            assistant_input_ids = tokenizer(features[0][f"assistant_{i}_text"], return_tensors="pt").input_ids
            assistant_audio_tokens = torch.tensor([features[0][f"assistant_{i}_codes"]], dtype=torch.int64)
            section_codes = torch.cat([start_token, user_input_ids, end_tokens, assistant_input_ids,final_tokens, assistant_audio_tokens, post_assistant_tokens], dim=1)
            my_input_ids = torch.cat([my_input_ids, section_codes], dim=1)
            audio_inputs.append(features[0][f"assistant_{i}_audio"]["array"])

    # pad audio_inputs and turn into tensor here
    max_len = max(len(a) for a in audio_inputs)

# 2. Pad each audio array with zeros up to max_len
    padded_audios = []
    for audio_array in audio_inputs:
        if len(audio_array) < max_len:
            pad_width = max_len - len(audio_array)
            # pad with zeros
            padded_array = np.concatenate(
                [audio_array, np.zeros(pad_width, dtype=audio_array.dtype)]
            )
        else:
            padded_array = audio_array
        padded_audios.append(padded_array)

    # 3. Stack everything into a single tensor of shape [batch_size, max_len]
    #    In your current code, it seems like you're only processing features[0],
    #    so effectively "batch_size" is 1 if you do not loop over all examples.
    audio_input = torch.tensor(padded_audios, dtype=torch.float32)


    return {
        "audio_values": audio_input.to(model.device).to(model.dtype),
        "input_ids": my_input_ids.to(model.device),
        "labels": my_input_ids.to(model.device),
        "attention_mask": torch.ones_like(audio_input).to(model.device)
    }




class AudioChatDataCollator:
    def __init__(self):
        self.greeting = "Hello world."

    def __call__(self, features):

        batch = inference_collator(features)

        return {
            "audio_values": batch["audio_values"].cpu(),
            "input_ids": batch["input_ids"].cpu(),
            "labels": batch["labels"].cpu(),
            "attention_mask": batch["attention_mask"].cpu()
        }


print("creating trainer")

training_args = TrainingArguments(
    output_dir="./hm_model-proj-2",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  # Changed to 16
    num_train_epochs=1,
    learning_rate=2e-3,  # Changed to 2*10^-3
    # save_strategy="no",
    logging_steps=1,
    evaluation_strategy="no",
    report_to="wandb",
    push_to_hub=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    save_steps=1000
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=AudioChatDataCollator(),
)

print("training")

trainer.train()


# Save model and tokenizer
output_dir = "mymodel_joint"
# trainer.save_model(output_dir)

trainer.model.save_pretrained(output_dir, safe_serialization=True)


# print("Loading the model using GazelleForConditionalGeneration directly")
# try:

#     loaded_model_custom = GazelleForConditionalGeneration.from_pretrained(
#         output_dir, config=special_config, new_vocab_size=True)
#     print("Loaded model with custom class:", loaded_model_custom)
# except Exception as e:
#     print("Error during model loading with GazelleForConditionalGeneration:", e)
