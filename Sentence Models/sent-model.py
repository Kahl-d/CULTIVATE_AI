import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

############################################
# 1. Define the Dataset
############################################

class SentenceDataset(Dataset):
    """
    Dataset for sentence-level classification.
    Assumes each row is a sentence with a corresponding label (0 or 1).
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        # Convert encoding to tensor
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label)
        return item

############################################
# 2. Define the Metrics Function
############################################

def compute_metrics(eval_pred):
    """
    Computes multiple metrics for binary classification (0,1):
      - Overall Accuracy
      - Binary Precision, Recall, F1 (pos_label=1)
      - Per-class (Class 0 & Class 1) metrics
      - Macro-average Precision, Recall, F1
      - Weighted-average Precision, Recall, F1
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Basic metrics with label=1 as the positive class
    accuracy = accuracy_score(labels, predictions)
    precision_bin = precision_score(labels, predictions, average="binary", pos_label=1)
    recall_bin = recall_score(labels, predictions, average="binary", pos_label=1)
    f1_bin = f1_score(labels, predictions, average="binary", pos_label=1)

    # Detailed breakdown using classification_report
    report_dict = classification_report(
        labels,
        predictions,
        output_dict=True,
        target_names=["Class 0", "Class 1"]
    )

    # Per-class metrics
    precision_class_0 = report_dict["Class 0"]["precision"]
    recall_class_0    = report_dict["Class 0"]["recall"]
    f1_class_0        = report_dict["Class 0"]["f1-score"]
    precision_class_1 = report_dict["Class 1"]["precision"]
    recall_class_1    = report_dict["Class 1"]["recall"]
    f1_class_1        = report_dict["Class 1"]["f1-score"]

    # Macro/Weighted averages
    macro_precision = report_dict["macro avg"]["precision"]
    macro_recall    = report_dict["macro avg"]["recall"]
    macro_f1        = report_dict["macro avg"]["f1-score"]
    weighted_precision = report_dict["weighted avg"]["precision"]
    weighted_recall    = report_dict["weighted avg"]["recall"]
    weighted_f1        = report_dict["weighted avg"]["f1-score"]

    # Return a consolidated dictionary for the Trainer
    return {
        "accuracy": accuracy,
        "precision_binary": precision_bin,
        "recall_binary": recall_bin,
        "f1_binary": f1_bin,
        "precision_class_0": precision_class_0,
        "recall_class_0": recall_class_0,
        "f1_class_0": f1_class_0,
        "precision_class_1": precision_class_1,
        "recall_class_1": recall_class_1,
        "f1_class_1": f1_class_1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

############################################
# 3. Main training, cross validation, and evaluation code
############################################

def main():
    # Model checkpoint
    model_checkpoint = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Load data (ensure your CSVs have columns: "sentence" and "label")
    train_df = pd.read_csv("aspirational_train.csv")
    test_df = pd.read_csv("aspirational_test.csv")
    
    train_texts = train_df["sentence"].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["sentence"].tolist()
    test_labels = test_df["label"].tolist()
    
    # A small hyperparameter grid. Feel free to expand or tweak these.
    hyperparam_grid = [
        {"learning_rate": 5e-5, "num_train_epochs": 3, "batch_size": 16},
        {"learning_rate": 3e-5, "num_train_epochs": 4, "batch_size": 16},
        {"learning_rate": 2e-5, "num_train_epochs": 3, "batch_size": 32},
    ]
    
    # 5-Fold cross validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    best_hp = None
    best_avg_f1 = -1.0
    
    print("\n=== Hyperparameter Tuning with Cross Validation ===")
    for combo_idx, combo in enumerate(hyperparam_grid, start=1):
        print(f"\n--- Hyperparameter Combo {combo_idx}/{len(hyperparam_grid)}: {combo} ---")
        fold_f1_scores = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(train_texts), start=1):
            print(f"\nFold {fold}/{n_splits}")
            # Prepare fold-specific data
            fold_train_texts = [train_texts[i] for i in train_index]
            fold_train_labels = [train_labels[i] for i in train_index]
            fold_val_texts = [train_texts[i] for i in val_index]
            fold_val_labels = [train_labels[i] for i in val_index]
            
            train_dataset = SentenceDataset(fold_train_texts, fold_train_labels, tokenizer)
            val_dataset = SentenceDataset(fold_val_texts, fold_val_labels, tokenizer)
            
            # Load fresh model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_checkpoint, num_labels=2
            )
            
            # Training arguments
            output_dir = f"./temp_fold_{fold}_combo_{combo_idx}"
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=combo["num_train_epochs"],
                per_device_train_batch_size=combo["batch_size"],
                per_device_eval_batch_size=combo["batch_size"],
                learning_rate=combo["learning_rate"],
                weight_decay=0.01,            # added weight decay for generalization
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1_binary",  # key in compute_metrics
                greater_is_better=True,
                logging_steps=50,
                report_to=[],
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
            )
            
            # Train and evaluate this fold
            trainer.train()
            fold_metrics = trainer.evaluate()
            
            # We'll use the binary F1 for picking the best model
            fold_f1 = fold_metrics.get("eval_f1_binary", 0.0)
            fold_f1_scores.append(fold_f1)
            
            print("Fold metrics:")
            for m_name, m_val in fold_metrics.items():
                print(f"  {m_name}: {m_val:.4f}")

        avg_f1 = np.mean(fold_f1_scores)
        print(f"\nAverage F1 (pos_label=1) for combo {combo}: {avg_f1:.4f}")
        
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_hp = combo
    
    print("\n=== Best Hyperparameter Combination ===")
    print(best_hp)
    print(f"Best average F1: {best_avg_f1:.4f}")
    
    ############################################
    # 4. Final Training on the Entire Training Set
    ############################################
    print("\n=== Final Training on Entire Training Set ===")
    final_train_dataset = SentenceDataset(train_texts, train_labels, tokenizer)
    final_model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2
    )
    
    final_output_dir = "./final_model"
    final_training_args = TrainingArguments(
        output_dir=final_output_dir,
        num_train_epochs=best_hp["num_train_epochs"],
        per_device_train_batch_size=best_hp["batch_size"],
        per_device_eval_batch_size=best_hp["batch_size"],
        learning_rate=best_hp["learning_rate"],
        weight_decay=0.01,     # keep same setting
        evaluation_strategy="no",  # No validation during final training
        logging_steps=100,
        report_to=[],
    )
    
    final_trainer = Trainer(
        model=final_model,
        args=final_training_args,
        train_dataset=final_train_dataset,
        compute_metrics=compute_metrics,  # (optional for logging)
    )
    
    final_trainer.train()
    final_trainer.save_model(final_output_dir)
    
    ############################################
    # 5. Evaluation on the Test Set
    ############################################
    print("\n=== Evaluating on Test Set ===")
    test_dataset = SentenceDataset(test_texts, test_labels, tokenizer)
    test_results = final_trainer.evaluate(test_dataset)

    print("\nTest Set Metrics (from Trainer.evaluate()):")
    for key, val in test_results.items():
        print(f"  {key}: {val:.4f}")
    
    # More detailed classification report & confusion matrix
    predictions_output = final_trainer.predict(test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=-1)

    # Classification report for final results
    print("\nDetailed Classification Report on Test Set:")
    print(classification_report(test_labels, preds, target_names=["Class 0", "Class 1"]))

    # Confusion matrix
    cm = confusion_matrix(test_labels, preds)
    print("\nConfusion Matrix (rows = true labels, cols = predicted labels):")
    print(cm)

if __name__ == "__main__":
    main()
