import sqlite3
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset

# --- Config ---
DB_PATH = '/home/fulian/RAG/data/request/sampled_sentences.db' 
MODEL_NAME = "/home/fulian/RAG/deberta-v3-small"
BASE_OUTPUT_DIR = "/home/fulian/RAG/src/training/checkpoint"
N_FOLDS = 5

# --- 1. Data Loading ---
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT sentence, before, after, response FROM sentences WHERE response IN ('KEEP', 'REMOVE')", conn)
conn.close()

# Map labels & Text
df['label'] = df['response'].map({'KEEP': 1, 'REMOVE': 0})
df['text'] = df['before'].fillna('') + " " + df['sentence'].fillna('') + " " + df['after'].fillna('')

# --- 2. K-Fold Setup ---
# Use StratifiedKFold to maintain label distribution (80% KEEP / 20% REMOVE) in every fold
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

# Store results across folds
fold_results = []

# --- 3. Main Training Loop ---
print(f"🚀 Starting {N_FOLDS}-Fold Cross Validation...")

# split() returns indices. We loop through them.
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")
    
    # A. Subset Data
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    # B. Convert to HF Datasets
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_val = val_ds.map(tokenize_function, batched=True)
    
    # C. Re-Initialize Model (CRITICAL: Must reset model every fold)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        id2label={0: "REMOVE", 1: "KEEP"},
        label2id={"REMOVE": 0, "KEEP": 1}
    )
    
    # D. Unique Output Dir per Fold
    fold_output_dir = f"{BASE_OUTPUT_DIR}/fold_{fold+1}"
    
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10, 
        
        # Eval strategy
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss", # Early stopping on Validation Loss (Recommended)
        greater_is_better=False,
        
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_strategy="epoch"
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # E. Train
    trainer.train()
    
    # F. Save & Log Metrics
    # We evaluate on the fold's specific validation set
    metrics = trainer.evaluate()
    fold_loss = metrics['eval_loss']
    fold_results.append(fold_loss)
    print(f"✅ Fold {fold+1} Validation Loss: {fold_loss:.4f}")
    
    # G. Plotting (Per Fold)
    def plot_fold_metrics(log_history, fold_num):
        train_steps = []
        train_loss = []
        eval_steps = []
        eval_loss = []
        
        for entry in log_history:
            if 'loss' in entry:
                train_steps.append(entry['step'])
                train_loss.append(entry['loss'])
            if 'eval_loss' in entry:
                eval_steps.append(entry['step'])
                eval_loss.append(entry['eval_loss'])
                
        plt.figure(figsize=(10, 5))
        plt.plot(train_steps, train_loss, label='Train Loss')
        if eval_loss:
            plt.plot(eval_steps, eval_loss, label='Val Loss', marker='o')
        
        plt.title(f'Fold {fold_num} Training Curves')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'fold_{fold_num}_metrics.png')
        plt.close() # Close plot to free memory
        
    plot_fold_metrics(trainer.state.log_history, fold + 1)
    
    # H. Cleanup (CRITICAL for Loop)
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

# --- 4. Final Summary ---
print("\n" + "="*30)
print("🏁 Cross Validation Complete")
print(f"Individual Folds (Eval Loss): {fold_results}")
print(f"🏆 Average Validation Loss: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")
print("="*30)