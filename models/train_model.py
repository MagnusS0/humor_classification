from transformers import TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Define constants
max_seq_length = 512
model_name = "unsloth/Phi-3-mini-4k-instruct"
load_in_4bit = True
dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float32

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply LoRA configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config={},
)

model.print_trainable_parameters()

# Setup tokenizer with chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# Load the data
data = pd.read_parquet("./humor_classification/data/processed/pretraining_data.parquet")

# Subsample to test the code
data = data.sample(frac=0.3, random_state=42)

# Make all columns objects
data["text"] = data["text"].astype("object")
data["label"] = data["label"].astype(int)

# Set up schema
schema = {
    "type": "object",
    "properties": {
        "rating": {
            "type": "number",
            "minimum": 0,
            "maximum": 4,
            "description": "The rating of the joke, from 0 to 4.",
        }
    },
}

# Make labels JSON format
data["label"] = data["label"].apply(lambda x: f'{{"rating": {x}}}')

# Set up prompt format
data["conversations"] = [
    [
        {
            "role": "system",
            "content": f"You are a joke evaluator that answers in JSON. Here's the json schema you must adhere to:\n{schema}",
        },
        {
            "role": "user",
            "content": f"""Your task is to evaluate jokes based on their funniness on a scale from 0 to 4, where 0 represents the least funny and 4 represents the most funny.\n "{joke}" """
        },
        {
            "role": "assistant", 
            "content": f"{label}"
        }
    ] for joke, label in zip(data["text"], data["label"])
]

# Split the data
train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True, stratify=data["label"])
test, val = train_test_split(test, test_size=0.5, random_state=42, stratify=test["label"])

# Convert to Dataset
train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(val)

# Format the prompts
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

# Adjust tokenizer settings
tokenizer.padding_side = 'right'

# Set up training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    max_grad_norm=0.3,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=1,
    optim="paged_adamw_32bit",
    weight_decay=0.001,
    lr_scheduler_type="constant",
    seed=3407,
    output_dir="outputs",
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=16,
    packing=False,
    args=training_args,
)

# Train the model
trainer_stats = trainer.train()

# Save the model
model.save_pretrained("lora_model_json_2")
