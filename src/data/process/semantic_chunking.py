import sqlite3
import os
import json
from tqdm import tqdm
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch

# --- Configuration ---
DB_PATH = '/home/fulian/RAG/data/request/inference_results_ensemble.db'
INPUT_FOLDER = '/home/fulian/RAG/data/processed'
OUTPUT_JSON_PATH = '/home/fulian/RAG/data/processed/semantic_chunks.json'
BATCH_SIZE = 20  # Number of files to process in one go

def get_unique_filenames_from_db(db_path):
    if not os.path.exists(db_path):
        return set()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT filename FROM inference_logs")
        ids = {row[0] for row in cursor.fetchall()}
        # print(ids[:5])  # Debug: print first 5 IDs
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()

    # unique_files = set()
    # for row_id in ids:
    #     if "_" in row_id:
    #         filename_part = row_id.rsplit("_", 1)[0]
    #         unique_files.add(filename_part)
    return list(ids) # Convert to list for batching

def main():
    # 1. Get List of Files
    print("🔍 Extracting filenames from database...")
    base_filenames = get_unique_filenames_from_db(DB_PATH)[:3]
    # print(base_filenames[:5])  # Print first 5 for verification
    print(f"📂 Found {len(base_filenames)} unique files to process.")
    
    if not base_filenames: return

    # 2. Setup LlamaIndex Semantic Splitter on GPU
    print("⏳ Loading Embedding Model on GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    embed_model = HuggingFaceEmbedding(
        model_name="/home/fulian/RAG/bge",
        device=device,           # <--- CRITICAL: Use GPU
        embed_batch_size=128,    # <--- CRITICAL: Larger internal batch size
        trust_remote_code=True
    )
    
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, 
        breakpoint_percentile_threshold=90, 
        embed_model=embed_model
    )

    all_chunks_data = []

    # 3. Process Files in Batches
    # This loop processes BATCH_SIZE files at a time to speed up embedding generation
    for i in tqdm(range(0, len(base_filenames), BATCH_SIZE), desc="Semantic Chunking"):
        batch_files = base_filenames[i : i + BATCH_SIZE]
        documents_batch = []
        file_metadata_batch = []

        # Prepare Batch
        for base_name in batch_files:
            file_path = os.path.join(INPUT_FOLDER, base_name)
            
            if not os.path.exists(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Store document and its ID metadata for later matching
                documents_batch.append(Document(text=text))
                file_metadata_batch.append(base_name)
            except Exception as e:
                print(f"Error reading {base_name}: {e}")

        if not documents_batch:
            print('⚠️ No valid documents found in this batch, skipping...')
            continue

        # Inference: Process entire batch at once
        # Note: Semantic splitter still iterates internally, but embedding model is now hot and batched
        try:
            nodes_list = splitter.get_nodes_from_documents(documents_batch)
        except Exception as e:
            print(f"❌ Error in batch {i}: {e}")
            continue

        for doc, fname in zip(documents_batch, file_metadata_batch):
            doc.metadata = {"filename": fname}

        # Re-run split with metadata enabled
        try:
            nodes_list = splitter.get_nodes_from_documents(documents_batch)
        except Exception as e:
            print(f"❌ Error in batch {i}: {e}")
            continue

        # Extract results
        for node in nodes_list:
            original_file = node.metadata.get("filename")
            clean_file_id = original_file.replace(".txt", "")
            
            pass # (Logic handled below in post-processing)

        # Optimization: Grouping logic
        from collections import defaultdict
        file_node_map = defaultdict(list)
        for node in nodes_list:
            fname = node.metadata["filename"]
            file_node_map[fname].append(node)
        
        for fname, nodes in file_node_map.items():
            clean_file_id = fname.replace(".txt", "")[5:]
            for idx, node in enumerate(nodes):
                chunk_entry = {
                    "id": f"{clean_file_id}_{idx}",
                    "text": node.get_content(),
                    "metadata": {
                        "original_file": fname,
                        "chunk_index": idx
                    }
                }
                all_chunks_data.append(chunk_entry)

    # 4. Save
    print(f"💾 Saving {len(all_chunks_data)} chunks to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_chunks_data, f, indent=4, ensure_ascii=False)
    
    print("✨ Done!")

if __name__ == "__main__":
    main()