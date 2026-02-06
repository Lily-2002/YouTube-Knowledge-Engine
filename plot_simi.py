import os
import glob
import random
import nltk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Ensure tokenizer is ready
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def analyze_consecutive_similarity(folder_path, num_files=50):
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('/home/fulian/RAG/bge', device=device)
    
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    if len(files) > num_files:
        files = random.sample(files, num_files)
    
    all_similarities = []

    print(f"Reading {len(files)} files and computing similarities...")

    # 2. Process Files
    for file_path in tqdm(files):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            if not text.strip(): continue

            sentences = nltk.sent_tokenize(text)
            if len(sentences) < 2: continue

            # Embed all sentences at once (Normalized by default)
            embeddings = model.encode(sentences, convert_to_numpy=True)

            # 3. Vectorized Cosine Similarity (Dot product of consecutive vectors)
            # Sim(i, i+1) = Vector_i . Vector_{i+1}
            # We multiply array[:-1] with array[1:] element-wise, then sum along axis 1
            sims = (embeddings[:-1] * embeddings[1:]).sum(axis=1)
            
            all_similarities.extend(sims.tolist())

        except Exception as e:
            print(f"Skipping {os.path.basename(file_path)}: {e}")

    # 4. Plotting
    if not all_similarities:
        print("No data found.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(all_similarities, bins=50, color='teal', alpha=0.7, edgecolor='black')
    plt.title(f"Cosine Similarity of Consecutive Sentences (n={len(all_similarities)})")
    plt.xlabel("Cosine Similarity (-1 to 1)")
    plt.ylabel("Frequency")
    plt.axvline(np.mean(all_similarities), color='red', linestyle='dashed', label=f'Mean: {np.mean(all_similarities):.2f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    save_path = "consecutive_similarity_dist.png"
    plt.savefig(save_path)
    print(f"Stats: Mean={np.mean(all_similarities):.3f}, Median={np.median(all_similarities):.3f}")
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    import torch # Imported here to check cuda inside function
    folder = "/home/fulian/RAG/data/processed" # <--- Update this
    analyze_consecutive_similarity(folder)