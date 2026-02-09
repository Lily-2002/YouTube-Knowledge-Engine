import os
import random
import torch
import numpy as np
import faiss
import json
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from collections import defaultdict

# --- Configuration ---
INPUT_FOLDER = '/home/fulian/RAG/data/processed'
OUTPUT_FOLDER = '/home/fulian/RAG/data/analysis'  # Where to save results
MODEL_PATH = "/home/fulian/RAG/Qwen_embed"
SAMPLE_SIZE = 50000   
NUM_CLUSTERS = 50     

# --- Setup ---
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("⏳ Loading Model...")
embed_model = HuggingFaceEmbedding(model_name=MODEL_PATH, device=device, trust_remote_code=True)

def get_random_sentences(folder, n=50000):
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]
    sampled_sentences = []
    file_prob = min(1.0, (n * 10) / 4000000) 
    
    print(f"📖 Sampling {n} sentences from {len(all_files)} files...")
    
    for fpath in all_files:
        if random.random() > file_prob and len(sampled_sentences) < n: continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()
                # Split by period, keep only sentences > 3 words
                sents = [s.strip() for s in text.split('.') if len(s.split()) > 3]
                sampled_sentences.extend(sents)
        except: continue
        if len(sampled_sentences) >= n: break
            
    return sampled_sentences[:n]

def main():
    # 1. Get Data
    sentences = get_random_sentences(INPUT_FOLDER, SAMPLE_SIZE)
    print(f"🧠 Embedding {len(sentences)} sentences...")
    
    embeddings = embed_model.get_text_embedding_batch(sentences)
    embeddings_np = np.array(embeddings).astype('float32')

    # 2. FAISS Clustering
    print(f"⚡ Clustering into {NUM_CLUSTERS} groups...")
    n_dims = embeddings_np.shape[1]
    kmeans = faiss.Kmeans(n_dims, NUM_CLUSTERS, niter=20, verbose=True, gpu=True)
    kmeans.train(embeddings_np)
    
    # Get the Centroids (The "Center" of each cluster)
    centroids = kmeans.centroids
    
    # Assign sentences
    D, I = kmeans.index.search(embeddings_np, 1)
    
    clusters = defaultdict(list)
    for sent_idx, cluster_idx in enumerate(I):
        clusters[int(cluster_idx[0])].append(sentences[sent_idx])

    # 3. Save Results
    print(f"💾 Saving results to {OUTPUT_FOLDER}...")
    
    # A. Save Centroids (The Vectors)
    np.save(os.path.join(OUTPUT_FOLDER, "cluster_centroids.npy"), centroids)
    
    # B. Save Readable Report
    report_path = os.path.join(OUTPUT_FOLDER, "cluster_report.txt")
    json_path = os.path.join(OUTPUT_FOLDER, "cluster_sentences.json")

    # Save full data to JSON (for debugging)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    # Write the readable report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== CLUSTERING REPORT ({len(sentences)} sentences) ===\n\n")
        
        for i in range(NUM_CLUSTERS):
            cluster_sents = clusters[i]
            if not cluster_sents: continue
            
            f.write(f"📁 CLUSTER {i:02d} | Size: {len(cluster_sents)}\n")
            f.write("-" * 40 + "\n")
            
            # Write up to 20 examples so you can really judge it
            examples = random.sample(cluster_sents, min(20, len(cluster_sents)))
            for ex in examples:
                f.write(f"   - {ex}\n")
            f.write("\n\n")

    print(f"✅ Done! Open '{report_path}' to inspect the clusters.")
    print(f"   If Cluster 5 is garbage, use 'centroids[5]' as your filter later.")

if __name__ == "__main__":
    main()