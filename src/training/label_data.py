import asyncio
import os
import sqlite3
import random
import time
from collections import defaultdict
import nltk
# Updated Import: Use AsyncOpenAI for async/await support
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIStatusError

# --- Configuration ---
# API_KEY = 'YOUR_KEY_HERE' 
# It is best practice to use environment variables
# API_KEY = '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'
API_KEY = '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'
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
DELAY_BETWEEN_REQUESTS = 3.0 
DAILY_TARGET = 20000
TOTAL_GOAL = 20000

if not API_KEY:
    print("❌ Error: API_KEY variable not set.")
    exit(1)

# --- Client Initialization ---
# FIX: Changed OpenAI to AsyncOpenAI to support 'await'
client = AsyncOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=API_KEY,
)

template = """Analyze the "Target Sentence" within its sequential "Context" to decide if it provides educational value.

### Task:
Label the Target Sentence as either "KEEP" (Instructional) or "REMOVE" (Noise).

### Classification Rules:
1. KEEP (Instructional) - High Priority:
   - Technical concepts, code, or subject matter.
   - Metaphors, analogies, or examples (e.g., "painkillers vs. vitamins").
   - Instructional transitions (e.g., "Think about it this way").
   - KEEP even if it contains fillers like "you know" or "like" IF the message is educational.

2. REMOVE (Noise) - Low Priority:
   - Administrative (e.g., "I'll upload slides").
   - Troubleshooting (e.g., "Can you hear me?").
   - Pure fillers with no meaning (e.g., "So, yeah...").

### Context:
- Before: {two_sentences_before}
- Target Sentence: "{TARGET_SENTENCE}"
- After: {two_sentences_after}

### Output:
Return ONLY the word "KEEP" or "REMOVE".
"""

instruction = 'You are an expert Educational Data Engineer.'


def get_sentences(file_path):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    with open(file_path, 'r', encoding='utf-8') as f:
        return nltk.sent_tokenize(f.read())


def get_existing_ids(db_path):
    if not os.path.exists(db_path):
        return set()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT ID FROM sentences WHERE response IS NOT NULL AND response != 'MISSING_RPD_LIMIT'")
        return {row[0] for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def delete_from_db(db_path, unique_id):
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM sentences WHERE ID = ?", (unique_id,))
        conn.commit()
    except Exception as e:
        print(f"⚠️ Warning: Could not delete ID {unique_id}: {e}")
    finally:
        conn.close()


def save_batch_to_db(db_path, results_list):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentences (
            ID TEXT PRIMARY KEY,
            sentence TEXT,
            before TEXT,
            after TEXT,
            response TEXT
        )
    ''')
    cursor.executemany(
        'INSERT OR REPLACE INTO sentences (id, sentence, before, after, response) VALUES (?, ?, ?, ?, ?)',
        results_list
    )
    conn.commit()
    conn.close()


def scan_and_sample(folder_path, n):
    print("🔍 Scanning files...")
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    file_counts = {}
    total_sentences = 0

    for filename in all_files:
        sents = get_sentences(os.path.join(folder_path, filename))
        if sents:
            file_counts[filename] = len(sents)
            total_sentences += len(sents)

    print(f"📊 Found {total_sentences} sentences total.")
    random.seed(42)

    universe = [(fname, i) for fname, count in file_counts.items() for i in range(count)]
    if n > len(universe):
        n = len(universe)

    selected_coords = random.sample(universe, n)

    grouped = defaultdict(list)
    for fname, idx in selected_coords:
        grouped[fname].append(idx)
    return grouped


def fetch_contexts_for_samples(folder_path, grouped_samples):
    final_data = []
    print("📖 Fetching contexts...")
    for filename, indices in grouped_samples.items():
        file_path = os.path.join(folder_path, filename)
        file_id = os.path.splitext(filename)[0][5:]
        sentences = get_sentences(file_path)

        for i in indices:
            before = " ".join(sentences[max(0, i - 2):i])
            after = " ".join(sentences[i + 1:min(len(sentences), i + 3)])
            target = sentences[i]

            prompt = template.format(
                two_sentences_before=before,
                TARGET_SENTENCE=target,
                two_sentences_after=after
            )

            final_data.append({
                "unique_id": f"{file_id}_{i}",
                "sentence": target,
                "before": before,
                "after": after,
                "prompt": prompt
            })
    return final_data


async def process_queue(data_items, db_path):
    success_count = 0

    print(f"🚀 Starting throttled processing for {len(data_items)} items...")
    print(f"⏱️  Speed limit: 1 request every {DELAY_BETWEEN_REQUESTS} seconds")
    print(f"📊 Expected rate: ~{60/DELAY_BETWEEN_REQUESTS:.1f} requests/minute")

    for i, item in enumerate(data_items):

        if success_count >= DAILY_TARGET:
            print("🛑 Daily target reached! Stopping for today.")
            break

        request_start = time.time()
        response_text = "ERROR"
        hit_limit = False

        try:
            # FIX: AsyncOpenAI allows this await call
            resp = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": item['prompt']}
                ],
                model=MODEL_NAME,
                max_tokens=10,
            )
            response_text = resp.choices[0].message.content.strip()
            success_count += 1

            if success_count % 10 == 0:
                print(f"   ✅ Progress: {success_count}/{DAILY_TARGET}")

        except RateLimitError as e:
            print(f"🚨 Rate Limit Hit (429)! {e}")
            print("   Saving current batch and stopping.")
            hit_limit = True
            response_text = "MISSING_RPD_LIMIT"
        except APIConnectionError as e:
            print(f"🔌 Connection error on {item['unique_id']}: {e}")
            response_text = "ERROR"
        except APIStatusError as e:
            print(f"❌ API error {e.status_code} on {item['unique_id']}: {e.response}")
            response_text = "ERROR"
        except Exception as e:
            print(f"❌ Unexpected error {item['unique_id']}: {type(e).__name__}: {e}")
            response_text = "ERROR"

        # Save result
        if response_text == "ERROR":
            delete_from_db(db_path, item["unique_id"])
        else:
            db_entry = [(
                item["unique_id"],
                item["sentence"],
                item["before"],
                item["after"],
                response_text
            )]
            save_batch_to_db(db_path, db_entry)

        if hit_limit:
            break

        # Async sleep handling
        elapsed = time.time() - request_start
        sleep_needed = DELAY_BETWEEN_REQUESTS - elapsed
        
        if sleep_needed > 0:
            await asyncio.sleep(sleep_needed)

    # FIX: AsyncOpenAI supports await close()
    await client.close()
    print(f"\n✨ Processing complete! Total successful: {success_count}")


def main_sampling_pipeline(folder_path, db_path, n_samples):
    grouped = scan_and_sample(folder_path, n_samples)
    all_data_items = fetch_contexts_for_samples(folder_path, grouped)

    processed_ids = get_existing_ids(db_path)
    print(f"♻️  Found {len(processed_ids)} items already completed.")

    items_to_process = [x for x in all_data_items if x['unique_id'] not in processed_ids]

    if not items_to_process:
        print("🎉 All items completed!")
        return

    print(f"📋 Items remaining: {len(items_to_process)}")

    asyncio.run(process_queue(items_to_process, db_path))


if __name__ == "__main__":
    folder_path = '/home/fulian/RAG/data/processed'
    db_path = '/home/fulian/RAG/data/request/sampled_sentences.db'

    main_sampling_pipeline(folder_path, db_path, TOTAL_GOAL)