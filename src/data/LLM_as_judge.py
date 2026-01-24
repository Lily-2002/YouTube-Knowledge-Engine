import os
import glob
import time
import nltk
from tqdm import tqdm
from huggingface_hub import InferenceClient

# Ensure NLTK tokenizer is ready
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CONFIGURATION ---
# FOLDER_PATH = "/home/fulian/RAG/data/processed"  # <--- CHANGE THIS
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

# Initialize Client
# Make sure your HF_TOKEN is set in your environment variables
client = InferenceClient(
    api_key="",
)

def is_sentence_meaningful(prev_sent, target_sent, next_sent):
    """
    Asks Llama 3.3 to judge the target_sent given its context.
    Returns: True (Keep) or False (Remove).
    """
    
    system_prompt = (
        "You are a strict editor for an academic transcript database. "
        "Your task is to identify and remove meaningless 'noise' sentences while keeping all informative content."
    )
    
    user_prompt = f"""
Evaluate the TARGET sentence below. Determine if it is informative/meaningful or if it is just noise (like stuttering, navigation text, housekeeping, or repetitive filler).

CONTEXT (Previous): "{prev_sent}"
TARGET (To Judge):  "{target_sent}"
CONTEXT (Next):     "{next_sent}"

Criteria for KEEP (Meaningful):
- It conveys facts, concepts, definitions, or clear thoughts.
- It is a coherent question or a bridge between ideas.

Criteria for DELETE (Noise):
- Navigation (e.g., "Go to next slide", "Page 4").
- Meta-commentary/Filler (e.g., "Can you hear me?", "Um, okay, so", "Copyright 2024").
- Broken fragments (e.g., "The the the").

ANSWER:
Reply with exactly one word: "KEEP" or "DELETE".
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # We don't use stream=True here because we need the full decision at once
            completion = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=10,  # We only need one word
                temperature=0.1 # Deterministic
            )
            
            response = completion.choices[0].message.content.strip().upper()
            
            # Simple parsing logic
            if "DELETE" in response:
                return False
            return True # Default to KEEP if it says "KEEP" or keeps talking
            
        except Exception as e:
            # Rate limit handling
            wait_time = 2 * (attempt + 1)
            # Only print if it's a real error, not just a small hiccup
            if attempt == max_retries - 1:
                print(f"⚠️ API Failed after retries: {e}")
            time.sleep(wait_time)

    # Fallback: If API fails completely, KEEP the sentence to be safe
    return True

def process_files_with_context(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    print(f"🚀 Starting Context-Aware Judge on {len(files)} files...")
    
    total_files_processed = 0
    
    for file_path in tqdm(files, desc="Files"):
        try:
            # 1. Read File
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            if not text.strip(): continue

            # 2. Split Sentences
            sentences = nltk.sent_tokenize(text)
            if not sentences: continue
            
            cleaned_sentences = []
            removed_count = 0
            
            # 3. Iterate One-by-One with Context
            total_sents = len(sentences)
            
            # Use nested tqdm to show progress per sentence inside the file
            # (Optional: remove this inner loop tqdm if it clutters your screen)
            for i in tqdm(range(total_sents), desc=f"Scanning {os.path.basename(file_path)}", leave=False):
                
                # Define Window
                prev_s = sentences[i-1] if i > 0 else "[START]"
                target_s = sentences[i]
                next_s = sentences[i+1] if i < total_sents - 1 else "[END]"
                
                # Call Judge
                keep = is_sentence_meaningful(prev_s, target_s, next_s)
                
                if keep:
                    cleaned_sentences.append(target_s)
                else:
                    removed_count += 1
            
            # 4. Save Changes
            # Only write if we actually processed the file
            if cleaned_sentences:
                new_text = " ".join(cleaned_sentences)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_text)
            
            total_files_processed += 1
            
            # Optional stats per file
            # print(f"Cleaned {os.path.basename(file_path)}: Removed {removed_count}/{total_sents} sentences.")

        except Exception as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {e}")

    print("\n" + "="*40)
    print(f"✅ PROCESSING COMPLETE")
    print(f"Total Files Processed: {total_files_processed}")
    print("="*40)

if __name__ == "__main__":
    process_files_with_context("/home/fulian/RAG/data/processed")