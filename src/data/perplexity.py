import os
import glob
import torch
import nltk
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Ensure NLTK tokenizer is ready
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 1. Load Model & Tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading GPT-2 on {device}...")
tokenizer = GPT2TokenizerFast.from_pretrained("/home/fulian/RAG/GPT2")
model = GPT2LMHeadModel.from_pretrained("/home/fulian/RAG/GPT2").to(device)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

def get_perplexity_batch(sentences, batch_size=32):
    """
    Computes PPL for a list of sentences efficiently using batches.
    Returns a list of float scores matching the order of input sentences.
    """
    perplexities = []
    
    # Process in chunks to prevent OOM (Out of Memory) errors
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            
            # Calculate loss per item manually
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss = loss.view(len(batch), -1)
            
            # Mask padding so it doesn't lower the perplexity artificialy
            mask = inputs["attention_mask"][..., 1:].contiguous()
            # Avoid division by zero
            sum_mask = mask.sum(dim=1)
            sum_mask[sum_mask == 0] = 1 
            
            seq_loss = (loss * mask).sum(dim=1) / sum_mask
            
            perplexities.extend(torch.exp(seq_loss).tolist())
            
    return perplexities

def clean_files_by_perplexity(folder_path, min_ppl=25, max_ppl=263):
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    print(f"📂 Found {len(files)} files. Starting sentence-level cleaning...")
    
    total_sentences_checked = 0
    total_sentences_removed = 0

    for file_path in tqdm(files, desc="Processing Files"):
        try:
            # 1. Read
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            if not text.strip(): continue

            # 2. Split into sentences
            sentences = nltk.sent_tokenize(text)
            if not sentences: continue
            
            total_sentences_checked += len(sentences)

            # 3. Compute Perplexity
            scores = get_perplexity_batch(sentences, batch_size=4)
            
            # 4. Filter 
            kept_sentences = []
            for sent, ppl in zip(sentences, scores):
                if min_ppl <= ppl <= max_ppl:
                    kept_sentences.append(sent)
            
            sentences_removed = len(sentences) - len(kept_sentences)
            total_sentences_removed += sentences_removed
            print(total_sentences_removed)
            # 5. Save back (Overwrite)
            # Only write if we actually have content left
            if kept_sentences:
                new_text = " ".join(kept_sentences)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_text)
            else:
                # Optional: If all sentences are deleted, you might want to empty the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("")

        except Exception as e:
            print(f"⚠️ Error in {os.path.basename(file_path)}: {e}")

    print("\n" + "="*40)
    print(f"✅ CLEANING COMPLETE")
    print(f"Sentences Checked: {total_sentences_checked}")
    print(f"Sentences Removed: {total_sentences_removed} ({(total_sentences_removed/total_sentences_checked)*100:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    # Set your folder path here
    target_folder = "/home/fulian/RAG/data/processed" 
    
    # Execute
    clean_files_by_perplexity(target_folder, min_ppl=25, max_ppl=263)