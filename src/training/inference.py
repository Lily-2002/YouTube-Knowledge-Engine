import os
import sqlite3
import torch
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
INPUT_FOLDER = '/home/fulian/RAG/data/processed' 
CHECKPOINT_BASE = "/home/fulian/RAG/src/training/checkpoint" 
DB_PATH = '/home/fulian/RAG/data/request/inference_results_ensemble.db'
NUM_FOLDS = 5
BATCH_SIZE = 128  # Increased for efficiency

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wrapper to treat the ensemble as a single model
class DebertaEnsemble(torch.nn.Module):
    def __init__(self, model_paths):
        super().__init__()
        self.models = torch.nn.ModuleList([
            AutoModelForSequenceClassification.from_pretrained(p) for p in model_paths
        ])

    def forward(self, **inputs):
        all_logits = [model(**inputs).logits for model in self.models]
        return torch.stack(all_logits).mean(dim=0)

print("⏳ Loading and Compiling Ensemble...")
fold_paths = [f"{CHECKPOINT_BASE}/fold_{i}" for i in range(1, NUM_FOLDS + 1)]
tokenizer = AutoTokenizer.from_pretrained(fold_paths[0])

ensemble = DebertaEnsemble(fold_paths).to(device)
ensemble.eval()

# Optional: Compile for speed (Linux/PyTorch 2.0+)
if hasattr(torch, 'compile'):
    try:
        ensemble = torch.compile(ensemble)
        print("🚀 Model compiled with torch.compile()!")
    except Exception as e:
        print(f"⚠️ Could not compile model: {e}")

# --- Database & Recovery Helper Functions ---

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    # Use WAL mode for better concurrency and safety
    conn.execute("PRAGMA journal_mode=WAL;") 
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inference_logs (
            ID TEXT PRIMARY KEY,
            filename TEXT,
            sentence_index INTEGER,
            sentence TEXT,
            context_before TEXT,
            context_after TEXT,
            model_prediction TEXT,
            avg_confidence_score REAL,
            vote_agreement REAL
        )
    ''')
    # Create an index on filename to speed up the recovery check
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON inference_logs (filename);')
    conn.commit()
    return conn

def get_processed_files(conn):
    """
    Returns a set of filenames that have already been fully processed and saved to the DB.
    """
    cursor = conn.cursor()
    try:
        # We assume if a filename exists in the logs, it was completed.
        # This works because we commit DB transactions per-file.
        cursor.execute("SELECT DISTINCT filename FROM inference_logs")
        processed = {row[0] for row in cursor.fetchall()}
        return processed
    except sqlite3.OperationalError:
        return set()

@torch.inference_mode()
def process_file_ensemble(filename, folder_path, conn):
    file_path = os.path.join(folder_path, filename)
    
    # Read text
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"❌ Error reading file {filename}: {e}")
        return 0, 0

    sentences = nltk.sent_tokenize(text)
    
    if not sentences:
        return 0, 0

    # 1. Prepare Inputs
    inputs_to_model = []
    metadata = []
    
    for i, target in enumerate(sentences):
        before = sentences[max(0, i-2):i]
        after = sentences[i+1:min(len(sentences), i+3)]
        context = f"{' '.join(before)} {target} {' '.join(after)}"
        
        inputs_to_model.append(context)
        metadata.append({
            'id': f"{filename}_{i}",
            'filename': filename,
            'index': i,
            'sentence': target,
            'before': " ".join(before),
            'after': " ".join(after)
        })

    # 2. Batch Inference
    predictions = []
    
    # FP16 Autocast for speed
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for i in range(0, len(inputs_to_model), BATCH_SIZE):
            batch_texts = inputs_to_model[i : i + BATCH_SIZE]
            
            encoded = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(device)
            
            avg_logits = ensemble(**encoded)
            probs = torch.softmax(avg_logits, dim=-1)
            
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            scores = torch.max(probs, dim=-1).values.cpu().numpy()
            
            for p, s in zip(preds, scores):
                predictions.append(("KEEP" if p == 1 else "REMOVE", float(s)))

    # 3. Save to DB (Atomic Transaction)
    cursor = conn.cursor()
    db_records = []
    kept_sentences = []
    
    for meta, (label, score) in zip(metadata, predictions):
        db_records.append((
            meta['id'],
            meta['filename'],
            meta['index'],
            meta['sentence'],
            meta['before'],
            meta['after'],
            label,
            score,
            0.0 
        ))
        
        if label == "KEEP":
            kept_sentences.append(meta['sentence'])
            
    # Insert logs
    cursor.executemany('''
        INSERT OR REPLACE INTO inference_logs 
        (ID, filename, sentence_index, sentence, context_before, context_after, model_prediction, avg_confidence_score, vote_agreement)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', db_records)
    
    # COMMIT happens here. If code crashes before this line, DB is empty for this file.
    conn.commit() 
    
    # 4. Rewrite File
    new_text = " ".join(kept_sentences)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_text)
        
    return len(sentences), len(kept_sentences)

# --- Main Execution ---
def main():
    conn = get_db_connection()
    
    # 1. Get all files
    all_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]
    print(f"📂 Found {len(all_files)} total files.")
    
    # 2. Check recovery (Which ones are already done?)
    print("🔍 Checking database for existing progress...")
    processed_files = get_processed_files(conn)
    print(f"♻️  Found {len(processed_files)} files already processed. Skipping them.")
    
    # 3. Filter the list
    files_to_process = [f for f in all_files if f not in processed_files]
    print(f"🚀 Starting processing on {len(files_to_process)} remaining files...")
    
    total_removed = 0
    
    # 4. Loop only through remaining files
    for filename in tqdm(files_to_process, desc="Ensemble Processing"):
        try:
            total_orig, total_kept = process_file_ensemble(filename, INPUT_FOLDER, conn)
            removed_count = total_orig - total_kept
            total_removed += removed_count
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            # If error occurs, we rollback DB just in case, though usually not needed due to transaction placement
            conn.rollback()
            
    conn.close()
    print("\n" + "="*30)
    print("✨ Processing Complete!")
    print(f"🗑️  Total sentences removed in this session: {total_removed}")
    print("="*30)

if __name__ == "__main__":
    main()