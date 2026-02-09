import os
import numpy as np
import json
import torch
import nltk
import glob
from tqdm import tqdm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Configuration ---
INPUT_FOLDER = '/kaggle/input/sentences/home/fulian/RAG/data/processed'
OUTPUT_FOLDER = '/kaggle/working/analysis'
MODEL_PATH = "/kaggle/input/qwen_embedding_0.6/pytorch/qwen_embedding_0.6/2"
PROGRESS_LOG = os.path.join(OUTPUT_FOLDER, "processed_files.log")
META_PATH = os.path.join(OUTPUT_FOLDER, "metadata.jsonl")

BATCH_SIZE = 128          
SAVE_EVERY_N = 50000     

# --- Setup ---
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
nltk.download('punkt', quiet=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⏳ Loading Model on {device}...")
embed_model = HuggingFaceEmbedding(model_name=MODEL_PATH, device=device, trust_remote_code=True)

def get_processed_files():
    if not os.path.exists(PROGRESS_LOG): return set()
    with open(PROGRESS_LOG, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

def get_next_chunk_id():
    existing_chunks = glob.glob(os.path.join(OUTPUT_FOLDER, "embeddings_part_*.npy"))
    if not existing_chunks: return 0
    ids = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in existing_chunks]
    return max(ids) + 1

def log_processed_files_batch(filenames):
    """Batch write filenames to log only when safe."""
    with open(PROGRESS_LOG, 'a', encoding='utf-8') as f:
        for name in filenames:
            f.write(name + "\n")

def stream_embeddings_to_disk():
    all_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")])
    processed_files = get_processed_files()
    files_to_process = [f for f in all_files if f not in processed_files]
    chunk_id = get_next_chunk_id()
    
    print(f"♻️  Skipping {len(processed_files)} files.")
    print(f"🚀 Processing {len(files_to_process)} files...")

    text_buffer = []      
    meta_buffer = []      
    chunk_embeddings = [] 
    chunk_metadata = []   
    

    files_fully_read_in_buffer = set() 

    for filename in tqdm(files_to_process, desc="Reading Files"):
        fpath = os.path.join(INPUT_FOLDER, filename)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()
            sentences = nltk.sent_tokenize(text)
            
            if not sentences:
                log_processed_files_batch([filename])
                continue

            for i, s in enumerate(sentences):
                text_buffer.append(s)
                meta_buffer.append({
                    "file": filename, "text": s, "index": i
                })

                # Inference Trigger
                if len(text_buffer) >= BATCH_SIZE:
                    embeddings = embed_model.get_text_embedding_batch(text_buffer)
                    chunk_embeddings.extend(embeddings)
                    chunk_metadata.extend(meta_buffer)
                    text_buffer = []
                    meta_buffer = []
                    
                # Disk Save Trigger
                if len(chunk_embeddings) >= SAVE_EVERY_N:
                    save_chunk(chunk_embeddings, chunk_metadata, chunk_id)
                    
                    
                    log_processed_files_batch(files_fully_read_in_buffer)
                    files_fully_read_in_buffer = set() 
                    
                    chunk_embeddings = [] 
                    chunk_metadata = []   
                    chunk_id += 1
            
            
            files_fully_read_in_buffer.add(filename)
        
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
            continue

    # Final Cleanup
    if text_buffer:
        embeddings = embed_model.get_text_embedding_batch(text_buffer)
        chunk_embeddings.extend(embeddings)
        chunk_metadata.extend(meta_buffer)
        
    if chunk_embeddings:
        save_chunk(chunk_embeddings, chunk_metadata, chunk_id)
        log_processed_files_batch(files_fully_read_in_buffer)

    print("\n✅ Embedding Complete!")

def save_chunk(embeddings, metadata, chunk_id):
    npy_path = os.path.join(OUTPUT_FOLDER, f"embeddings_part_{chunk_id:03d}.npy")
    np.save(npy_path, np.array(embeddings).astype('float16'))
    
    with open(META_PATH, 'a', encoding='utf-8') as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    stream_embeddings_to_disk()