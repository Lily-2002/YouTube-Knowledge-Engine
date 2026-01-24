import os
import glob
import nltk
from tqdm import tqdm

# Ensure the sentence tokenizer data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def count_total_sentences(folder_path):
    # Find all .txt files
    file_list = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not file_list:
        print("❌ No .txt files found in the specified folder.")
        return

    total_sentences = 0
    total_files = len(file_list)
    
    print(f"📖 Scanning {total_files} files for sentences...")

    # Process every file
    for file_path in tqdm(file_list, desc="Processing Files"):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                
                if not text.strip():
                    continue
                
                # sent_tokenize handles periods, question marks, and exclamation points
                # even if there are no newlines (\n) in the text.
                sentences = nltk.sent_tokenize(text)
                total_sentences += len(sentences)
                
        except Exception as e:
            print(f"⚠️ Error reading {os.path.basename(file_path)}: {e}")

    print("\n" + "="*30)
    print(f"📊 FINAL STATISTICS")
    print("="*30)
    print(f"Total Files Scanned:    {total_files}")
    print(f"Total Sentences Found:  {total_sentences:,}") # Uses comma as thousands separator
    print("="*30)

if __name__ == "__main__":
    # Update this to your specific directory
    target_folder = "/home/fulian/RAG/data/processed" 
    count_total_sentences(target_folder)