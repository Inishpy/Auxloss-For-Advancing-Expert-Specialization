import os
# Reduce CUDA allocator fragmentation before any CUDA calls
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer
)
import argparse
import datetime
import torch
from datasets import Dataset, DatasetDict
import pandas as pd

# ── CLI arguments ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fine-tune a causal LM on a single flan_task file with LoRA.")
parser.add_argument("--model_name_or_path", type=str,
                    default="deepseek-ai/deepseek-moe-16b-chat")
parser.add_argument("--dataset_name", type=str, required=True,
                    help="Path to a flan_task JSON/JSONL file with 'inputs' and 'targets' columns.")
parser.add_argument("--output_prefix", type=str, default="checkpoints")
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.1)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                    help="Keep at 1 for 80 GB GPUs with this model; use gradient_accumulation_steps to scale effective batch.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                    help="Effective batch = per_device_train_batch_size * gradient_accumulation_steps.")
parser.add_argument("--num_train_epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--logging_steps", type=int, default=10)
parser.add_argument("--save_steps", type=int, default=500)
parser.add_argument("--save_total_limit", type=int, default=5)
parser.add_argument("--ref", type=str, default="flan")
args = parser.parse_args()

MODEL_NAME = args.model_name_or_path
REF        = args.ref

# ── Derive task name from dataset filename ───────────────────────────────────
filename = os.path.basename(args.dataset_name)
task_name = filename.split(":")[0] if ":" in filename else os.path.splitext(filename)[0]

timestamp       = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
time_stamp_path = f"./timestamp_{task_name}"
logging_dir     = f"./Output/Tensorboard/{MODEL_NAME}/{REF}/{task_name}/{timestamp}"
output_dir      = f"{args.output_prefix}/{task_name}"
saved_model_dir = f"./SavedModels/{MODEL_NAME}/{REF}/{task_name}/saved_model_{timestamp}"

with open(time_stamp_path, "w") as f:
    f.write(timestamp + "\n")
    f.write(logging_dir + "\n")

print(f"Task          : {task_name}")
print(f"Timestamp     : {timestamp}")
print(f"Logging dir   : {logging_dir}")
print(f"Output dir    : {output_dir}")

# ── Load tokenizer & model ───────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# ── Gradient checkpointing ───────────────────────────────────────────────────
# Recomputes activations during the backward pass instead of storing them.
# Cuts activation memory from O(n_layers * seq * batch) to O(sqrt), at the
# cost of ~30% extra compute.  Essential for a 16B MoE model on 80 GB.
model.gradient_checkpointing_enable()
# gradient_checkpointing requires input embeddings to propagate gradients
model.enable_input_require_grads()

# ── Load flan_task dataset ───────────────────────────────────────────────────
df = pd.read_json(args.dataset_name, lines=True)

col_map = {}
for col in df.columns:
    if col.lower() in ("inputs", "input"):
        col_map[col] = "inputs"
    elif col.lower() in ("targets", "target", "outputs", "output"):
        col_map[col] = "targets"
if col_map:
    df = df.rename(columns=col_map)

if "inputs" not in df.columns or "targets" not in df.columns:
    raise ValueError(
        f"Dataset file '{args.dataset_name}' must contain 'inputs' and 'targets' columns. "
        f"Found: {list(df.columns)}"
    )

split_idx = int(len(df) * 0.8)
ds = DatasetDict({
    "train":      Dataset.from_pandas(df.iloc[:split_idx].reset_index(drop=True)),
    "validation": Dataset.from_pandas(df.iloc[split_idx:].reset_index(drop=True)),
})
print(f"Dataset splits -- train: {len(ds['train'])}, validation: {len(ds['validation'])}")

# ── LoRA configuration ───────────────────────────────────────────────────────
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
)
model = get_peft_model(model, lora_config)

# FIX: DeepSeek MoE routing buffer and PEFT adapter weights are float32 by
# default; cast everything to bfloat16 to match the loaded base model.
model = model.to(torch.bfloat16)

model.print_trainable_parameters()

# ── Tokenisation ─────────────────────────────────────────────────────────────
def process_func(example):
    MAX_LENGTH = args.max_seq_length
    instruction = tokenizer(
        f"<|start_header_id|>user<|end_header_id|>\n\n{example['inputs']}"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['targets']}<|eot_id|>", add_special_tokens=False)

    input_ids      = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels         = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids      = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels         = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

tokenized_ds = ds.map(process_func, remove_columns=ds["train"].column_names)

# ── Training arguments ───────────────────────────────────────────────────────
# Memory budget on an 80 GB GPU with DeepSeek-MoE-16B (bfloat16):
#   Model weights       ~34 GB
#   Optimizer states    ~17 GB  (8-bit paged AdamW, ~0.5x model)
#   Activations         ~4 GB   (gradient checkpointing enabled)
#   Gradient buffers    ~4 GB
#   --------------------------------
#   Total               ~59 GB  (leaves ~20 GB headroom)
#
# With standard AdamW (fp32 states) optimizer states alone would be ~68 GB,
# which combined with weights would already exceed 80 GB.
#
# bf16=True is intentionally omitted -- the model is already fully bfloat16
# from the explicit .to(torch.bfloat16) cast above; enabling autocast on top
# of an already-bfloat16 model re-introduces float32<->bfloat16 casts inside
# Trainer that cause the MoE routing dtype mismatch.
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    warmup_steps=args.warmup_steps,
    weight_decay=0.01,
    learning_rate=args.learning_rate,
    logging_dir=logging_dir,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="tensorboard",
    # 8-bit paged AdamW: cuts optimizer states from ~68 GB (fp32 Adam)
    # to ~17 GB while keeping training stability. Requires bitsandbytes.
    optim="paged_adamw_8bit",
    # Disable Trainer's own gradient checkpointing flag; we already called
    # model.gradient_checkpointing_enable() directly above.
    gradient_checkpointing=False,
)
print("logging_dir:", training_args.logging_dir)
print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")

# ── Trainer ──────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
trainer.save_model(saved_model_dir)
print(f"Model saved to: {saved_model_dir}")