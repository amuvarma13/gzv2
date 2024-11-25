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

# 1. Register the custom model and config **at the very top**
MODEL_FOR_CAUSAL_LM_MAPPING.register("gazelle", GazelleForConditionalGeneration)
CONFIG_MAPPING["gazelle"] = GazelleConfig  # Register GazelleConfig

# 2. Device and dtype setup
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

# 3. Load tokenizer first
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# 4. Add custom tokens
number_add_tokens = 6 * 1024 + 10  # 6144 + 10 = 6154
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]  # 6155 tokens
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})
print("Tokenizer vocab size after adding tokens:", len(tokenizer))

# 5. Initialize Weights & Biases
wandb.init(
    project="colab-a100-40gb",
    name="vmllamaspeechcontentonly-500k-8h100s-3"
)

# 6. Load expressions
file_path = 'transcribe_exps.txt'
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        expressions = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"The file {file_path} does not exist.")
except IOError:
    print(f"An error occurred while reading the file {file_path}.")

# 7. Load dataset
dsn = "amuvarma/mls-eng-10k-dev-3k"
ds = load_dataset(dsn, split="dev")
dataset = ds

# 8. Initialize configuration with updated vocab_size
model_id = "amuvarma/convo-tts-tune-7contentonly"

config = GazelleConfig(
    audio_model_id="facebook/wav2vec2-base-960h", 
    text_model_id=model_id, 
    audio_token_index=134411, 
    vocab_size=len(tokenizer),  # Updated vocab_size
)

# Diagnostic print
print("Config model_type:", config.model_type)  # Should print 'gazelle'
print("Config vocab_size:", config.vocab_size)  # Should match tokenizer's vocab_size

# 9. Instantiate the model with the updated config
model = GazelleForConditionalGeneration(config).to(dtype=dtype)



# 10. Resize token embeddings to match tokenizer's vocab size
model.resize_token_embeddings(len(tokenizer))
# print("Model's embed_tokens.weight shape:", model.language_model.model.embed_tokens.weight.shape)

special_config =  model.config

# 11. Freeze all parameters except 'multi_modal_projector'
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "multi_modal_projector" in name:
        param.requires_grad = True
        torch.nn.init.normal_(param, mean=0.0, std=0.02)

# 12. Verify trainable parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name} - {param.shape}")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable_params:,}")
print(f"All parameters: {all_params:,}")

# 13. Initialize audio processor
audio_processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

print("Creating collator")

# 14. Define inference collator
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

print("Creating data collator")

# 15. Define custom data collator
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

print("Creating trainer")

# 16. Define training arguments
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

# 17. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=AudioChatDataCollator(),
)

print("Training")

# Uncomment to start training
# trainer.train()

# 18. Save model and tokenizer
output_dir = "mymodel"
print(trainer.model)

trainer.model.save_pretrained(output_dir, safe_serialization=True)
# tokenizer.save_pretrained(output_dir)  # It's good practice to save the tokenizer as well

# 19. Load the model using AutoModelForCausalLM
print("Loading the model using AutoModelForCausalLM")
try:
    config_loaded = AutoConfig.from_pretrained(output_dir)
    loaded_model = AutoModelForCausalLM.from_pretrained(output_dir, config=config_loaded)
    print("Loaded model:", loaded_model)
except Exception as e:
    print("Error during model loading with AutoModelForCausalLM:", e)

# 20. (Optional) Load the model using the custom class directly
print("Loading the model using GazelleForConditionalGeneration directly")
try:

    loaded_model_custom = GazelleForConditionalGeneration.from_pretrained(output_dir, config=special_config, new_vocab_size=True)
    print("Loaded model with custom class:", loaded_model_custom)
except Exception as e:
    print("Error during model loading with GazelleForConditionalGeneration:", e)
