import random
from datasets import load_dataset
import wandb
from datasets import Dataset

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

from transformers import Wav2Vec2Config, LlamaConfig
import torch
import transformers
from transformers import Trainer, TrainingArguments
import torchaudio
from transformers import CONFIG_MAPPING

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

number_processes = 8
batch_size = 8

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
        
        # Check if loss exists in logs
        if "loss" not in logs:
            return  # Skip logging if there's no loss
            
        if global_step % 2 == 0:
            wandb.log({"text_loss": logs["loss"], "step": global_step})
        else:
            wandb.log({"audio_loss": logs["loss"], "step": global_step})


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
# output_dir = "amuvarma/e2e-1"
# output_dir = "amuvarma/snac-2m-proj-qa-speechqa-14374"
output_dir = "hm_model-proj-2/checkpoint-2943"
# print(model)


print("before loading")

model = GazelleForConditionalGeneration.from_pretrained(output_dir, config=special_config, new_vocab_size=True)

print("after loading")

# print(model)

# dsn = "amuvarma/voice-assistant-200k-processed-1"
dsn1 = "amuvarma/va-330k-380k-snac-StTtS"
# dsn = "amuvarma/mls-eng-10k-dev-3k"
ds1 = load_dataset(dsn1, split="train")
ds1 = ds1.select(range(0, 20000))

#round length to nearest multiple of 64
def round_length_to_64(dataset):
    new_length = (len(dataset) // 64) * 64
    return dataset.select(range(new_length))



dsn2 = "amuvarma/voice-assistant-250-300k-processed"
ds2 = load_dataset(dsn2, split="train")
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

def remove_long_audio(dataset, max_seconds=60.0):
    indices_to_keep = []

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        audio = example['question_audio']
        duration = len(audio['array']) / audio['sampling_rate']
        if max_seconds >= duration:
            indices_to_keep.append(i)

    filtered_dataset = dataset.select(indices_to_keep)

    return filtered_dataset
ds1 = remove_short_audio(ds1)

ds1 = ds1.filter(lambda example: len(example["snac_tokens"]) < 2700)
ds1 = remove_long_audio(ds1)

ds2 = remove_short_audio(ds2)
ds2 = remove_long_audio(ds2)

ds1 = round_length_to_64(ds1)
ds2 = round_length_to_64(ds2)

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
            return self.dataset1[dataset_index]
        else:
            dataset_index = super_batch * self.batch_total + (position_in_super_batch - self.batch_total)
            return self.dataset2[dataset_index]

class AlternatingDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)




for param in model.parameters():
    param.requires_grad = False

special_config = model.config
wandb.init(
    project="projection-layer-2",
    name="r12-12"
)

file_path = 'transcribe_exps.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        expressions = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"The file {file_path} does not exist.")
except IOError:
    print(f"An error occurred while reading the file {file_path}.")




batch_total = number_processes * batch_size
train_dataset = BatchedAlternatingDataset(ds1, ds2, batch_total)


model = model.to(dtype=dtype)


for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "multi_modal_projector" in name:
        param.requires_grad = True
    # if "language_model" in name:
    #     param.requires_grad = True

trainable_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

def inference_collator(audio_input, user_res, ass_res, content_tokens):

    user_input_ids = tokenizer(user_res, return_tensors="pt").input_ids
    assistant_input_ids = tokenizer(ass_res, return_tensors="pt").input_ids


    start_of_system = torch.tensor([[128256+8]], dtype=torch.int64)
    end_of_system = torch.tensor([[128256+9]], dtype=torch.int64)
    end_of_text = torch.tensor([[128009]], dtype=torch.int64)

    content_tensor = torch.tensor([content_tokens], dtype=torch.int64)

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


    true_labels = torch.full_like(labels, -100)
    true_labels[:, user_tokens.shape[1]:] = labels[:, user_tokens.shape[1]:]



    attention_mask = torch.ones_like(labels)

    return {
        "audio_values": audio_input.to(model.device).to(model.dtype),
        "input_ids": labels.to(model.device),
        "labels": true_labels.to(model.device),
        "attention_mask": attention_mask.to(model.device)
    }




class AudioChatDataCollator:
    def __init__(self):
        self.greeting = "Hello world."

    def __call__(self, features):
        audio = torch.tensor([features[0]["question_audio"]["array"]])
        assistant_response = features[0]["answer"]
        user_response = "<|audio|>"
        content_tokens = []
        if "snac_tokens" in features[0]:
            content_tokens = features[0]["snac_tokens"]
        # content_tokens = []

        batch = inference_collator(audio, user_response, assistant_response, content_tokens)

        return {
            "audio_values": batch["audio_values"].cpu(),
            "input_ids": batch["input_ids"].cpu(),
            "labels": batch["labels"].cpu(),
            "attention_mask": batch["attention_mask"].cpu()
        }


print("creating trainer")

training_args = TrainingArguments(
    output_dir="./hm_model-proj-3",
    per_device_train_batch_size=batch_size,
    # gradient_accumulation_steps=8, 
    num_train_epochs=1,
    learning_rate=2e-4,  # Changed to 2*10^-3
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

trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=AudioChatDataCollator(),
)

print("training")

trainer.train()


output_dir = "mymodel_joint"

trainer.model.save_pretrained(output_dir, safe_serialization=True)

