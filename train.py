import random
from datasets import load_dataset
import wandb
from datasets import Dataset


from transformers import Wav2Vec2Config, LlamaConfig
import torch
import transformers
from transformers import Trainer, TrainingArguments
import torchaudio
from transformers import CONFIG_MAPPING

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING


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


model_id = "./mymodel"
config = GazelleConfig(
    audio_model_id="facebook/wav2vec2-base-960h",
    text_model_id=model_id,
    audio_token_index=156939,
    vocab_size=156939,

)

model_id = "./mymodel"
config = GazelleConfig(
    audio_model_id="facebook/wav2vec2-base-960h",
    text_model_id=model_id,
    audio_token_index=156939,  # Updated audio_token_index
    vocab_size=156939,  # Updated vocab_size
)
model = GazelleForConditionalGeneration(config).to(dtype=dtype)
special_config =  model.config
# output_dir = "models/checkpoint-78"
output_dir = "checkpoints/checkpoint-490"
model = GazelleForConditionalGeneration.from_pretrained(output_dir, config=special_config, new_vocab_size=True)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct")
number_add_tokens = 7 * 4096 + 10
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})

print(model)
model.resize_token_embeddings(len(tokenizer))



special_config = model.config



wandb.init(
    project="projection-layer",
    name = "wav2vec-2linear-zuck"
    )

file_path = 'transcribe_exps.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        expressions = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"The file {file_path} does not exist.")
except IOError:
    print(f"An error occurred while reading the file {file_path}.")


dsn = "amuvarma/snacced-flat-zuck-convo-sttsed"
# dsn = "amuvarma/mls-eng-10k-dev-3k"
ds = load_dataset(dsn, split="train")

dataset = ds

# dataset = dataset.select(range(0, 500))


model = model.to(dtype=dtype)

print(model)
# First freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Then unfreeze just the multi_modal_projector
# First set requires_grad
for name, param in model.named_parameters():
    if "multi_modal_projector" in name:
        param.requires_grad = True

# Print to verify
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name} - {param.shape}")

trainable_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

print(f"\nTrainable parameters: {trainable_params:,}")
print(f"All parameters: {all_params:,}")

audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)


print("creating collator")


def inference_collator(audio_input, user_res, ass_res):

    user_input_ids = tokenizer(user_res, return_tensors="pt").input_ids
    assistant_input_ids = tokenizer(ass_res, return_tensors="pt").input_ids

    # print("user_input_ids", user_input_ids.shape)

    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    start_of_system = torch.tensor([[128256+8]], dtype=torch.int64)
    end_of_system = torch.tensor([[128256+9]], dtype=torch.int64)
    end_of_text = torch.tensor([[128009]], dtype=torch.int64)

    system_message = "You are an AI assistant who will answer the user's questions and follow the user's instructions."
    system_input_ids = tokenizer(system_message, return_tensors="pt").input_ids
    system_tokens = torch.cat(
        [start_of_system, system_input_ids, end_of_text, end_of_system],  dim=1)

    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)
    final_tokens = torch.tensor([[128009]], dtype=torch.int64)

    user_tokens = torch.cat(
        [system_tokens, start_token, user_input_ids, end_tokens], dim=1)

    labels = torch.cat([system_tokens, start_token, user_input_ids, end_tokens,
                       assistant_input_ids, final_tokens], dim=1)

    true_labels = torch.full_like(labels, -100)
    true_labels[:, user_tokens.shape[1]:] = labels[:, user_tokens.shape[1]:]

    # print("true_labels", true_labels)
    # print("input_ids", labels)

    attention_mask = torch.ones_like(labels)

    return {
        "audio_values": audio_input.to(model.device).to(model.dtype),
        "input_ids": labels.to(model.device),
        "labels": true_labels.to(model.device),
        "attention_mask": attention_mask.to(model.device)
    }


print("creating data collator")


class AudioChatDataCollator:
    def __init__(self):
        self.greeting = "Hello world."

    def __call__(self, features):
        audio = torch.tensor([features[0]["answer_audio"]["array"]])
        assistant_response = features[0]["answer"]
        # user_response = features[0]["user"]

        user_response = "<|audio|>"

        batch = inference_collator(audio, user_response, assistant_response)

        return {
            "audio_values": batch["audio_values"].cpu(),
            "input_ids": batch["input_ids"].cpu(),
            "labels": batch["labels"].cpu(),
            "attention_mask": batch["attention_mask"].cpu()
        }


print("creating trainer")

training_args = TrainingArguments(
    output_dir="./modelssnac",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # Changed to 16
    num_train_epochs=1,
    learning_rate=2e-5,  # Changed to 2*10^-3
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
    save_steps=15000
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=AudioChatDataCollator(),
)

print("training")

trainer.train()

