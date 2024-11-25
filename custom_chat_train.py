from datasets import Dataset, load_dataset
from transformers import (
    Wav2Vec2Config, 
    LlamaConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig, 
    Trainer, 
    TrainingArguments, 
    Wav2Vec2Processor, 
    CONFIG_MAPPING
)
import torch
import transformers
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from gzf import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)
import wandb
import torchaudio
import random

# Register the custom model and config
MODEL_FOR_CAUSAL_LM_MAPPING.register("gazelle", GazelleForConditionalGeneration)
CONFIG_MAPPING["gazelle"] = GazelleConfig  # Register GazelleConfig

# Device and dtype setup
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

# Model and tokenizer setup
model_id = "amuvarma/convo-tts-tune-7contentonly"
config = GazelleConfig(
    audio_model_id="facebook/wav2vec2-base-960h", 
    text_model_id=model_id, 
    audio_token_index=134411, 
    vocab_size=134411,
)

model = GazelleForConditionalGeneration(config).to(dtype=dtype)
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Add custom tokens
number_add_tokens = 6 * 1024 + 10
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})
print("model device", model.device)
model.resize_token_embeddings(len(tokenizer))

# Initialize Weights & Biases
wandb.init(
    project="colab-a100-40gb",
    name="vmllamaspeechcontentonly-500k-8h100s-3"
)

# Load expressions
file_path = 'transcribe_exps.txt'
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        expressions = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"The file {file_path} does not exist.")
except IOError:
    print(f"An error occurred while reading the file {file_path}.")

# Load dataset
dsn = "amuvarma/mls-eng-10k-dev-3k"
ds = load_dataset(dsn, split="dev")
dataset = ds

# Freeze all parameters except 'multi_modal_projector'
model = model.to(dtype=dtype)
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "multi_modal_projector" in name:
        param.requires_grad = True
        torch.nn.init.normal_(param, mean=0.0, std=0.02)

# Verify trainable parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name} - {param.shape}")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable_params:,}")
print(f"All parameters: {all_params:,}")

# Initialize audio processor
audio_processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

print("creating collator")

# Define inference collator
def inference_collator(audio_input, ass_res, instruction="Transcribe the following \n<|audio|>"):
    audio_values = audio_processor(
        audio=audio_input, return_tensors="pt", sampling_rate=16000
    ).input_values

    user_phrase = "<|audio|>"
    user_input_ids = tokenizer(user_phrase, return_tensors="pt").input_ids
    assistant_input_ids = tokenizer(ass_res, return_tensors="pt").input_ids

    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)
    final_tokens = torch.tensor([[128009]], dtype=torch.int64)

    user_tokens = torch.cat([start_token, user_input_ids, end_tokens], dim=1)
    labels = torch.cat([start_token, user_input_ids, end_tokens, assistant_input_ids, final_tokens], dim=1)
    true_labels = torch.full_like(labels, -100)
    true_labels[:, user_tokens.shape[1]:] = labels[:, user_tokens.shape[1]:]
    attention_mask = torch.ones_like(labels)

    return {
        "audio_values": audio_input.to(model.device).to(model.dtype),
        "input_ids": labels.to(model.device),
        "labels": true_labels.to(model.device),
        "attention_mask": attention_mask.to(model.device)
    }

print("creating data collator")

# Define custom data collator
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

print("creating trainer")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=2e-3,
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

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=AudioChatDataCollator(),
)

print("training")

# Uncomment to start training
# trainer.train()

# Save model and tokenizer
output_dir = "mymodel"
print(trainer.model)

trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)  # It's good practice to save the tokenizer as well

# Load the model using AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained(output_dir)
loaded_model = AutoModelForCausalLM.from_pretrained(output_dir, config=config)
print("Loaded model:", loaded_model)
