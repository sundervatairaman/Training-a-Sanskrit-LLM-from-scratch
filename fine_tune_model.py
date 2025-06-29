import os
import re
import math
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from torch.nn import Module


# Configuration
model_name = "./linked_model_with_custom_tokenizer"
output_dir = "./fine_tuned_model_output_1"

os.makedirs(output_dir, exist_ok=True)

# Step 1: Verify GPU availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure you have a GPU and CUDA installed.")

device = torch.device("cuda")
print(f"Using device: {device}")

# Step 2: Load model in full precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

# Step 3: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Step 4: Load and preprocess training data
if not os.path.exists("combined_output.txt"):
    raise FileNotFoundError("The training data file 'combined_output.txt' was not found.")

with open("combined_output.txt", "r", encoding="utf-8") as f:
    text_data = [line.strip() for line in f.readlines() if line.strip()]

if not text_data:
    raise ValueError("The training data file is empty or contains no valid text.")

dataset = Dataset.from_dict({"text": text_data})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        max_length=512,
        truncation=True,
        padding="max_length",
        return_special_tokens_mask=True
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 5: Split into train and validation
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Step 6: Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 7: Define perplexity calculator

import math
from transformers import TrainerCallback

from transformers import Trainer

class PerplexityTrainer(Trainer):
    def on_evaluate(self, args, state, control, **kwargs):
        metrics = super().evaluate(**kwargs)
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            try:
                perplexity = math.exp(eval_loss)
            except:
                perplexity = float("inf")
            print(f"Step {state.global_step} | Validation Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
        return control


class EvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, data_collator, device):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.device = device

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        model.eval()
        total_loss = 0
        total_length = 0

        # Simulate small batches
        for i in range(0, len(self.eval_dataset), 4):
            batch = self.eval_dataset[i:i+4]  # Get a slice of examples
            batch = self.data_collator(batch)  # Convert to tensor batch
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_length += batch['input_ids'].size(0)

        avg_loss = total_loss / total_length
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float("inf")

        print(f"Epoch {state.epoch}: Validation Loss = {avg_loss:.4f}, Perplexity = {perplexity:.2f}")
        return control

# Step 8: Configure training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=5,
    gradient_accumulation_steps=5,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=100,
    fp16=True,
    optim="adamw_torch",
    save_strategy="steps",
    save_steps=10000,
    evaluation_strategy="steps",
    eval_steps=1000,
    report_to="none",
    gradient_checkpointing=True,
    label_names=["labels"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Step 9: Create trainer
trainer = PerplexityTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 10: Train and save best model
print("Starting training...")
trainer.train()

# Save final model (best one already loaded if `load_best_model_at_end=True`)
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {os.path.abspath(output_dir)}")