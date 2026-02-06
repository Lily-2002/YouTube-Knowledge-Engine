from data import summarize_batch
from data import run_batch_inference
import json
import re
from pathlib import Path
from tqdm import tqdm
import sqlite3

def create_sqlite_table(db_path, table_name, columns):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    table_creation_query = f"""
    CREATE TABLE {table_name} (
        ID TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        TimeStamp REAL NOT NULL,
        Summary TEXT,
        Questions TEXT
    );
"""
    cursor.execute(table_creation_query)
    print("Table is Ready")
    cursor.close()

def insert_sqlite_record(db_path, table_name, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = []
    for idx, unit in enumerate(data):
        unit_id = f"{file_path.stem}_{idx + 1}"
        unit['id'] = unit_id
        texts.append(unit['text'])
    summarys = 

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
def split_questions(question_text):
    """
    Splits a single string containing multiple questions into a list of strings.
    Example: "How old are you? Where are you from" -> ["How old are you?", "Where are you from?"]
    """
    if not question_text:
        return []
    raw_questions = re.split(r'(?<=[?!.])\s+', str(question_text))
    cleaned_questions = [q.strip() for q in raw_questions if q.strip()]
    if not cleaned_questions and question_text.strip():
        return [question_text.strip()]
        
    return cleaned_questions
def process_units(folder_path):
    base_path = Path(folder_path)
    json_files = list(base_path.glob("*.json"))
    
    if not json_files:
        print("No JSON files found.")
        return
    all_texts = []
    reference_map = []

    print(f"📖 Reading {len(json_files)} JSON files...")
    files_cache = {} 

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                print(f"⚠️ Warning: {file_path.name} root is a dict, expected list of units. Skipping.")
                continue
                
            files_cache[file_path] = data
            
            for idx, unit in enumerate(data):
                unit_id = f"{file_path.stem}_{idx + 1}"
                unit['id'] = unit_id
                text_content = unit.get('text', "")
                if text_content:
                    all_texts.append(text_content)
                    reference_map.append({
                        "file_path": file_path,
                        "unit_idx": idx
                    })
                    
        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")

    if not all_texts:
        print("No text fields found to process.")
        return

    print(f"✅ Extracted {len(all_texts)} text units. Starting Batch AI Processing...")

    print("⏳ Generating Summaries...")
    summaries_list = summarize_batch(all_texts)
    
    # B. Generate Questions
    print("⏳ Generating Questions...")
    questions_list = run_batch_inference(all_texts)

    # Sanity Check
    if len(summaries_list) != len(all_texts) or len(questions_list) != len(all_texts):
        print("❌ CRITICAL ERROR: Output list length does not match input list length!")
        return

    # --- STEP 3: MERGE & SAVE ---
    print("💾 Merging results and saving files...")

    for i, ref in enumerate(reference_map):
        f_path = ref['file_path']
        u_idx = ref['unit_idx']
        
        # Retrieve the original unit from our cache
        unit = files_cache[f_path][u_idx]
        
        # 1. Add Summary
        unit['summary'] = summaries_list[i]
        
        # 2. Add Questions (Split string into list)
        raw_q_string = questions_list[i]
        unit['questions'] = split_questions(raw_q_string)

    # Write files back to disk
    for file_path, data in files_cache.items():
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"   Saved: {file_path.name}")
        except Exception as e:
            print(f"   ❌ Error saving {file_path.name}: {e}")

    print("\n🎉 All processing complete!")
if __name__ == '__main__':
    process_units('/home/fulian/RAG/data/Timestamp')