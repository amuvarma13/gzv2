import random
from datasets import load_dataset
import wandb
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import Wav2Vec2Config, LlamaConfig
import torch
import transformers
from transformers import Trainer, TrainingArguments
import torchaudio
from transformers import CONFIG_MAPPING

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

# dsn1 = "amuvarma/voice-assistant-250-255k-processed"
dsn1 = "amuvarma/voice-assistant-250-300k-processed"
dsn2 = "amuvarma/26k-stts-duplex-convos-raw-fac"
# dsn1 = "amuvarma/10k-stts-duplex-convos-raw-fac-1dups-contentonly"
# dsn2 = "amuvarma/10k-audio-audio-contentonly"
ds1 = load_dataset(dsn1, split="train")
def remove_short_audio(dataset, min_seconds=1.0):
    indices_to_keep = []

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        audio = example['question_audio']
        duration = len(audio['array']) / audio['sampling_rate']
        if duration >= min_seconds:
            indices_to_keep.append(i)

    filtered_dataset = dataset.select(indices_to_keep)

    return filtered_dataset

filtered_ds = remove_short_audio(ds1)

ds2 = load_dataset(dsn2, split="train")

from gzf import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)

batch_size = 8
number_processes = 8

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
number_add_tokens = 6 * 1024 + 10  # 6144 + 10 = 6154
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]  # 6155 tokens
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})


model_id = "amuvarma/convo-tts-tune-7contentonly"
config = GazelleConfig(
    audio_model_id="facebook/wav2vec2-base-960h",
    text_model_id=model_id,
    audio_token_index=134411,
    vocab_size=len(tokenizer),  # Updated vocab_size
)
model = GazelleForConditionalGeneration(config).to(dtype=dtype)
special_config =  model.config
# output_dir = "models/checkpoint-78"
output_dir = "amuvarma/llm-train-tune-voiceassistant-checkpoint-3098"
model = GazelleForConditionalGeneration.from_pretrained(output_dir, config=special_config, new_vocab_size=True)

for param in model.parameters():
    param.requires_grad = False

special_config = model.config
wandb.init(
    project="gazelle-tune-llm",
    name="r7-12"
)

file_path = 'transcribe_exps.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        expressions = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"The file {file_path} does not exist.")
except IOError:
    print(f"An error occurred while reading the file {file_path}.")






class BatchedAlternatingDataset(Dataset):
    def __init__(self, dataset1, dataset2, batch_total):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_total = batch_total
        self.length = 2 * min(len(dataset1), len(dataset2))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        super_batch = index // (2 * self.batch_total)
        
        position_in_super_batch = index % (2 * self.batch_total)
        
        if position_in_super_batch < self.batch_total:
            dataset_index = super_batch * self.batch_total + position_in_super_batch
            # print(f"returning from dataset1: {dataset_index}")
            return self.dataset1[dataset_index]
        else:
            dataset_index = super_batch * self.batch_total + (position_in_super_batch - self.batch_total)
            # print(f"returning from dataset2: {dataset_index}")
            return self.dataset2[dataset_index]

class AlternatingDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


class FSDPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        sampler = AlternatingDistributedSampler(
            self.train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False, 
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def log(self, logs):
        super().log(logs)
        global_step = self.state.global_step
        if global_step % 2 == 0:
            wandb.log({"text_loss": logs["loss"], "step": global_step})
        else:
            wandb.log({"audio_loss": logs["loss"], "step": global_step})

batch_total = number_processes * batch_size
train_dataset = BatchedAlternatingDataset(filtered_ds, filtered_ds, batch_total)






print(train_dataset)


model = model.to(dtype=dtype)

# First freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Then unfreeze just the multi_modal_projector
# First set requires_grad
# for name, param in model.named_parameters():
    # if "multi_modal_projector" in name:
    #     param.requires_grad = True
    # if "language_model" in name:
    #     param.requires_grad = True

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


def inference_collator(audio_input, user_res, ass_res, content_tokens):

    user_input_ids = tokenizer(user_res, return_tensors="pt").input_ids
    assistant_input_ids = tokenizer(ass_res, return_tensors="pt").input_ids

    # print("user_input_ids", user_input_ids.shape)

    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    start_of_system = torch.tensor([[128256+8]], dtype=torch.int64)
    end_of_system = torch.tensor([[128256+9]], dtype=torch.int64)
    end_of_text = torch.tensor([[128009]], dtype=torch.int64)

    content_tensor = torch.tensor([content_tokens], dtype=torch.int64)
    content_tensor = content_tensor + 128266

    system_message = "You are an AI assistant who will answer the user's questions and follow the user's instructions."
    system_input_ids = tokenizer(system_message, return_tensors="pt").input_ids
    system_tokens = torch.cat(
        [start_of_system, system_input_ids, end_of_text, end_of_system],  dim=1)

    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)
    final_tokens = torch.tensor([[128009, 128257 ]], dtype=torch.int64)
    post_assistant_tokens = torch.tensor([[128258, 128262]])

    user_tokens = torch.cat(
        [system_tokens, start_token, user_input_ids, end_tokens], dim=1)


    
    if len(content_tokens):
            labels = torch.cat([system_tokens, start_token, user_input_ids, end_tokens,
                      assistant_input_ids, final_tokens, content_tensor, post_assistant_tokens], dim=1)
            
    else:
        labels = torch.cat([system_tokens, start_token, user_input_ids, end_tokens,
                      assistant_input_ids, final_tokens], dim=1)

    # labels = torch.cat([system_tokens, start_token, user_input_ids, end_tokens,
    #                    assistant_input_ids, final_tokens], dim=1)

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


#************************************************************
# def inference_collator(audio_input, user_res, ass_res):

#     user_input_ids = tokenizer(user_res, return_tensors="pt").input_ids
#     assistant_input_ids = tokenizer(ass_res, return_tensors="pt").input_ids

#     # print("user_input_ids", user_input_ids.shape)

#     # input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     start_of_system = torch.tensor([[128256+8]], dtype=torch.int64)
#     end_of_system = torch.tensor([[128256+9]], dtype=torch.int64)
#     end_of_text = torch.tensor([[128009]], dtype=torch.int64)

#     system_message = "You are an AI assistant who will answer the user's questions and follow the user's instructions."
#     system_input_ids = tokenizer(system_message, return_tensors="pt").input_ids
#     system_tokens = torch.cat(
#         [start_of_system, system_input_ids, end_of_text, end_of_system],  dim=1)

#     start_token = torch.tensor([[128259]], dtype=torch.int64)
#     end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)
#     final_tokens = torch.tensor([[128009]], dtype=torch.int64)

#     user_tokens = torch.cat(
#         [system_tokens, start_token, user_input_ids, end_tokens], dim=1)

#     labels = torch.cat([system_tokens, start_token, user_input_ids, end_tokens,
#                        assistant_input_ids, final_tokens], dim=1)

#     true_labels = torch.full_like(labels, -100)
#     true_labels[:, user_tokens.shape[1]:] = labels[:, user_tokens.shape[1]:]

#     # print("true_labels", true_labels)
#     # print("input_ids", labels)

#     attention_mask = torch.ones_like(labels)

#     return {
#         "audio_values": audio_input.to(model.device).to(model.dtype),
#         "input_ids": labels.to(model.device),
#         "labels": true_labels.to(model.device),
#         "attention_mask": attention_mask.to(model.device)
#     }

#************************************************************

print("creating data collator")


class AudioChatDataCollator:
    def __init__(self):
        self.greeting = "Hello world."

    def __call__(self, features):
        audio = torch.tensor([features[0]["question_audio"]["array"]])
        assistant_response = features[0]["answer"]
        user_response = "<|audio|>"
        content_tokens = []
        if("facodec_1" in features[0]):
            content_tokens = features[0]["facodec_1"]

        batch = inference_collator(audio, user_response, assistant_response, content_tokens)

        return {
            "audio_values": batch["audio_values"].cpu(),
            "input_ids": batch["input_ids"].cpu(),
            "labels": batch["labels"].cpu(),
            "attention_mask": batch["attention_mask"].cpu()
        }


print("creating trainer")

training_args = TrainingArguments(
    output_dir="./modelsjoint",
    per_device_train_batch_size=batch_size,
    # gradient_accumulation_steps=,  # Changed to 16
    num_train_epochs=1,
    learning_rate=2e-5,  # Changed to 2*10^-3
    # save_strategy="no",
    logging_steps=1,
    evaluation_strategy="no",
    report_to="wandb",
    push_to_hub=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    # warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    save_steps=1000
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=AudioChatDataCollator(),
)

print("training")

trainer.train()


# Save model and tokenizer
output_dir = "mymodel_joint"
# trainer.save_model(output_dir)

from pydub import AudioSegment
sound = AudioSegment.from_mp3("3.mp3")
sound.export("recorded_audio.wav", format="wav")
import torchaudio


test_audio, sr = torchaudio.load("recorded_audio.wav")
print(test_audio.shape, sr)

if sr != 16000:
    print("resampling audio")
    test_audio = torchaudio.transforms.Resample(sr, 16000)(test_audio)
test_audio = test_audio[0]
print("new", test_audio.shape)

audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)
audio_values = audio_processor(
    audio=test_audio, return_tensors="pt", sampling_rate=16000
).input_values
def new_inference_collator():
    user_phrase = "<|audio|>" #<|audio|>"
    user_input_ids = tokenizer(user_phrase, return_tensors="pt").input_ids
    end_of_text = torch.tensor([[128009]], dtype=torch.int64)
    start_of_system = torch.tensor([[128256+8]], dtype=torch.int64)
    end_of_system = torch.tensor([[128256+9]], dtype=torch.int64)

    system_message = "You are an AI assistant who will answer the user's questions."
    system_input_ids = tokenizer(system_message, return_tensors="pt").input_ids
    system_tokens = torch.cat(
        [start_of_system, system_input_ids, end_of_text, end_of_system],  dim=1)

    # print("user_input_ids", user_input_ids.shape)

    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)
    final_tokens = torch.tensor([[128009]], dtype=torch.int64)


    user_tokens = torch.cat(
        [system_tokens, start_token, user_input_ids, end_tokens], dim=1)

    return user_tokens
user_tokens = new_inference_collator()
loaded_model_custom = trainer.model
myinputs= {
  "audio_values": audio_values.to(loaded_model_custom.device).to(loaded_model_custom.dtype),
  "input_ids": user_tokens.to(loaded_model_custom.device),
  # "input_ids": tokenizer("Okay, so what would be a healthier breakfast option then? Can you tell me?", return_tensors="pt").input_ids.to("cuda")
}

loaded_model_custom.eval()
outs = loaded_model_custom.generate(
    **myinputs,
    max_new_tokens=100,
    temperature=0.5,
    repetition_penalty=1.1,
    top_p=0.8,
    eos_token_id=128262,
    )

import torch

def extract_tokens_after_value(tensor, target_start=128257, target_end=128258):
    tensor_list = tensor.tolist()

    start_index = tensor_list.index(target_start)
    try:
        end_index = tensor_list.index(target_end, start_index)
        return tensor_list[start_index + 1:end_index]
    except ValueError:
        return tensor_list[start_index + 1:]

text_tokens = extract_tokens_after_value(outs[0], 128261, 128257)
dec = tokenizer.decode(text_tokens)
print("Decoded:", dec)

# trainer.model.save_pretrained(output_dir, safe_serialization=True)


# print("Loading the model using GazelleForConditionalGeneration directly")
# try:

#     loaded_model_custom = GazelleForConditionalGeneration.from_pretrained(
#         output_dir, config=special_config, new_vocab_size=True)
#     print("Loaded model with custom class:", loaded_model_custom)
# except Exception as e:
#     print("Error during model loading with GazelleForConditionalGeneration:", e)
