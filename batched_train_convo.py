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


from gzf import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)

batch_size = 1
number_processes = 2

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
    audio_token_index=156939,  # Updated audio_token_index
    vocab_size=156939,  # Updated vocab_size
)
model = GazelleForConditionalGeneration(config).to(dtype=dtype)
special_config =  model.config
# output_dir = "models/checkpoint-78"
output_dir = "modelsaltbatch/checkpoint-2974"
model = GazelleForConditionalGeneration.from_pretrained(output_dir, config=special_config, new_vocab_size=True)

for param in model.parameters():
    param.requires_grad = False

special_config = model.config
wandb.init(
    project="gazelle-tune",
    name="r0-4-1"
)


print(model)

dsn1 = "amuvarma/snacced-zuck-convo-sttsed-proc-no-discrepancies"
# dsn2 = "amuvarma/snacced-zuck-convo-sttsed-proc-no-discrepancies"
# dsn1 = "amuvarma/qa_pairs_regular-sttsed-proc"
dsn2 = "amuvarma/qa_pairs_regular-sttsed-proc"

# dsn1 = "amuvarma/zuckqa-with-qaaudio-total-cast-snacced"
# dsn2  = "amuvarma/zuckqa-with-qaaudio-total-cast-snacced"


ds1 = load_dataset(dsn1, split="train")
ds2 = load_dataset(dsn2, split="train")

# def remove_short_audio(dataset, min_seconds=1.0):
#     indices_to_keep = []

#     for i, example in tqdm(enumerate(dataset), total=len(dataset)):
#         audio = example['question_audio']
#         duration = len(audio['array']) / audio['sampling_rate']
#         if duration >= min_seconds:
#             indices_to_keep.append(i)

#     filtered_dataset = dataset.select(indices_to_keep)

#     return filtered_dataset

# filtered_ds1 = remove_short_audio(ds1)
# filtered_ds2 = remove_short_audio(ds2)


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
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
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
    def log(self, logs, extra):
        super().log(logs)
        global_step = self.state.global_step
        
        # Check if loss exists in logs
        if "loss" not in logs:
            return  # Skip logging if there's no loss
            
        if global_step % 2 == 0:
            wandb.log({"text_loss": logs["loss"], "step": global_step})
        else:
            wandb.log({"audio_loss": logs["loss"], "step": global_step})

batch_total = number_processes * batch_size
train_dataset = BatchedAlternatingDataset(ds1, ds2, batch_total)


print(train_dataset)


model = model.to(dtype=dtype)

for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "multi_modal_projector" in name:
        param.requires_grad = True
    # if "language_model" in name:
    #     param.requires_grad = True


audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)



class AudioChatDataCollator:
    def __init__(self):
        self.greeting = "Hello world."

    def __call__(self, features):
        # Find the length of the longest sequence
        max_len = max(len(seq) for seq in features[0]["audios"])

        # Create padded sequences
        padded_sequences = []
        for seq in features[0]["audios"]:
            padded = seq + [0] * (max_len - len(seq))  # Zero padding
            padded_sequences.append(padded)

        # Convert to tensor
        audios = torch.tensor(padded_sequences)
        input_ids = torch.tensor([features[0]["input_ids"]], dtype=torch.int64)
        attention_mask = torch.ones_like(input_ids)

        return {
            "audio_values": audios.cpu(),
            "input_ids": input_ids.cpu(),
            "labels": input_ids.cpu(),
            "attention_mask": attention_mask.cpu()
        }


print("creating trainer")

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=batch_size,
    num_train_epochs=1,
    learning_rate=2e-6, 
    logging_steps=1,
    evaluation_strategy="no",
    report_to="wandb",
    push_to_hub=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    lr_scheduler_type="cosine",
    bf16=True,
    save_steps=490
)

trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=AudioChatDataCollator(),
)

print("training")

trainer.train()
