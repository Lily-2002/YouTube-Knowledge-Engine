import os
import glob
import random
import torch
import nltk
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Ensure NLTK tokenizer is ready
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_model_and_tokenizer():
    """Load GPT-2 Small model efficiently."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GPT-2 on {device}...")
    tokenizer = GPT2TokenizerFast.from_pretrained("/home/fulian/RAG/GPT2")
    model = GPT2LMHeadModel.from_pretrained("/home/fulian/RAG/GPT2").to(device)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device

def read_and_sample_sentences(folder_path, sample_size=5000):
    """
    Reads .txt files from folder, splits into sentences, 
    and returns a random sample of 5000.
    """
    all_sentences = []
    file_list = glob.glob(os.path.join(folder_path, "*.txt"))
    
    print(f"📖 Reading files from: {folder_path}")
    
    # Shuffle file list to get a random distribution even if we stop early
    random.shuffle(file_list)
    
    for file_path in tqdm(file_list, desc="Loading Files"):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                if not text.strip(): continue
                
                # Split text into sentences using NLTK (more robust than regex)
                sents = nltk.sent_tokenize(text)
                
                # Filter out tiny snippets (e.g. "Fig 1.")
                sents = [s for s in sents if len(s.split()) > 3]
                all_sentences.extend(sents)
                
                # Optimization: If we have WAY more than needed, we can stop reading
                # to save RAM (e.g., stop at 100k candidates)
                if len(all_sentences) > 100000:
                    break
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Found {len(all_sentences)} candidate sentences.")
    
    if len(all_sentences) > sample_size:
        return random.sample(all_sentences, sample_size)
    return all_sentences

def compute_perplexity(sentences, model, tokenizer, device, batch_size=8):
    """
    Computes PPL for a list of sentences using batch processing.
    """
    perplexities = []
    print(f"🧠 Computing perplexity for {len(sentences)} sentences...")

    for i in tqdm(range(0, len(sentences), batch_size), desc="Inference"):
        batch = sentences[i:i + batch_size]
        
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            # Calculate loss per item manually to avoid averaging over padding
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss = loss.view(len(batch), -1)
            # Mask padding
            mask = inputs["attention_mask"][..., 1:].contiguous()
            seq_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)
            
            perplexities.extend(torch.exp(seq_loss).tolist())
            
    return perplexities

def plot_distribution(ppl_scores):
    """
    Plots the histogram of perplexity scores with percentiles.
    """
    # Remove extreme outliers for plotting clarity (e.g., PPL > 1000)
    filtered_scores = [s for s in ppl_scores if s < 600]
    outlier_count = len(ppl_scores) - len(filtered_scores)
    
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.hist(filtered_scores, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Percentiles
    p5 = np.percentile(ppl_scores, 5)
    p10 = np.percentile(ppl_scores, 10)
    p90 = np.percentile(ppl_scores, 90)
    
    # Add vertical lines
    plt.axvline(p5, color='red', linestyle='dashed', linewidth=1.5, label=f'5th % (Too Low?): {p5:.1f}')
    plt.axvline(p10, color='orange', linestyle='dashed', linewidth=1.5, label=f'10th % (Safe Low): {p10:.1f}')
    plt.axvline(p90, color='green', linestyle='dashed', linewidth=1.5, label=f'90th % (Safe High): {p90:.1f}')
    
    plt.title(f"Perplexity Distribution (n={len(ppl_scores)}) - {outlier_count} outliers > 600 hidden")
    plt.xlabel("Perplexity Score (Lower = More Predictable)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save the plot
    save_path = "perplexity_distribution.png"
    plt.savefig(save_path)
    print(f"📊 Plot saved to: {save_path}")
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    folder = "/home/fulian/RAG/data/processed"  # <--- CHANGE THIS
    
    # 1. Setup
    model, tokenizer, device = get_model_and_tokenizer()
    
    # 2. Get Data
    sampled_sentences = read_and_sample_sentences(folder, sample_size=5000)
    
    if sampled_sentences:
        # 3. Compute
        scores = compute_perplexity(sampled_sentences, model, tokenizer, device)
        
        # 4. Visualize
        plot_distribution(scores)
        
        # 5. Print Stats
        print("\n--- Statistics ---")
        print(f"Min PPL: {min(scores):.2f}")
        print(f"Max PPL: {max(scores):.2f}")
        print(f"Mean PPL: {np.mean(scores):.2f}")
        print(f"Median PPL: {np.median(scores):.2f}")