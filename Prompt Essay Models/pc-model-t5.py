# Extended script: K-fold cross validation + hyperparameter tuning + final best training
# This approach helps find the best hyperparameters among a small grid, using cross validation.

import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import KFold

############################################
# 1. Define dataset class
############################################

class CCTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.samples = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        example = self.samples[idx]
        prompt_text = example['prompt']
        completion_text = example['completion']

        # tokenize prompt as input, completion as target
        tokenized_input = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True
        )
        tokenized_output = self.tokenizer(
            completion_text,
            max_length=self.max_length,
            truncation=True
        )

        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']
        labels = tokenized_output['input_ids']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

############################################
# 2. Load the data from JSONL
############################################

data_file = "aspirational_train2.jsonl"  # or your combined dataset

all_data = []
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        all_data.append(entry)

# Shuffle
random.shuffle(all_data)

############################################
# 3. Prepare base model checkpoint + tokenizer
############################################

model_checkpoint = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

############################################
# 4. Define a hyperparameter grid
############################################

hyperparam_grid = [
    {"learning_rate": 1e-4, "num_train_epochs": 3, "batch_size": 2},
    {"learning_rate": 5e-5, "num_train_epochs": 4, "batch_size": 2},
    {"learning_rate": 3e-5, "num_train_epochs": 5, "batch_size": 4},
]

############################################
# 5. K-Fold cross validation for hyperparam combos
############################################

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

def compute_metrics(eval_pred):
    return {}

best_hp = None
best_loss = float('inf')

for combo_idx, combo in enumerate(hyperparam_grid, start=1):
    print(f"\n=== Hyperparam Combo {combo_idx}/{len(hyperparam_grid)} ===")
    print(combo)

    fold_losses = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data), start=1):
        print(f"  Starting fold {fold}...")

        # Partition the data
        train_fold = [all_data[i] for i in train_idx]
        val_fold = [all_data[i] for i in val_idx]

        # Create datasets
        train_dataset_fold = CCTDataset(train_fold, tokenizer)
        val_dataset_fold = CCTDataset(val_fold, tokenizer)

        # Model + collator
        model_fold = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        data_collator_fold = DataCollatorForSeq2Seq(tokenizer, model=model_fold)

        # Create training args for this fold + HP combo
        training_args_fold = TrainingArguments(
            output_dir="./temp_ckpts_fold",  # overwritten each fold
            run_name=f"fold_{fold}_combo_{combo_idx}",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            save_total_limit=1,
            report_to=[],
            learning_rate=combo["learning_rate"],
            num_train_epochs=combo["num_train_epochs"],
            per_device_train_batch_size=combo["batch_size"],
            per_device_eval_batch_size=combo["batch_size"],
        )

        trainer_fold = Trainer(
            model=model_fold,
            args=training_args_fold,
            train_dataset=train_dataset_fold,
            eval_dataset=val_dataset_fold,
            data_collator=data_collator_fold,
            compute_metrics=compute_metrics,
        )

        trainer_fold.train()
        eval_metrics = trainer_fold.evaluate()
        loss_val = eval_metrics.get("eval_loss", float('inf'))
        fold_losses.append(loss_val)
        print(f"    Fold {fold} eval_loss = {loss_val}")

    # Average fold loss for this combo
    avg_combo_loss = np.mean(fold_losses)
    print(f"  => Average cross-fold loss = {avg_combo_loss}")

    if avg_combo_loss < best_loss:
        best_loss = avg_combo_loss
        best_hp = combo

print("\n=== BEST HYPERPARAM COMBO ===")
print(best_hp)
print("Cross-fold eval_loss:", best_loss)

############################################
# 6. Final training on entire dataset
############################################

print("\n=== Final Training with Best Hyperparams ===")

# Re-init model
model_final = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
final_collator = DataCollatorForSeq2Seq(tokenizer, model=model_final)

# Build a dataset over all_data if you want full coverage
final_dataset = CCTDataset(all_data, tokenizer)

final_args = TrainingArguments(
    output_dir="./cct_finetuning",
    run_name="cct_finetuning_final",
    eval_strategy="no",
    save_strategy="no",
    logging_steps=100,
    report_to=[],
    learning_rate=best_hp["learning_rate"],
    num_train_epochs=best_hp["num_train_epochs"],
    per_device_train_batch_size=best_hp["batch_size"],
    per_device_eval_batch_size=best_hp["batch_size"],
)

trainer_final = Trainer(
    model=model_final,
    args=final_args,
    train_dataset=final_dataset,
    data_collator=final_collator,
    compute_metrics=compute_metrics,
)

trainer_final.train()

trainer_final.save_model("./models/pc-aspirational-model-2")

print("Final model saved with best hyperparams.")
