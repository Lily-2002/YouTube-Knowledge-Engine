import sqlite3
import os
import nltk
from tqdm import tqdm

# --- Config ---
INPUT_FOLDER = '/home/fulian/RAG/data/processed'
DB_PATH = '/home/fulian/RAG/data/request/inference_results_ensemble.db'

def verify_and_fix_consistency():
    print("🕵️Starting Consistency Check (DB vs File System)...")
    
    if not os.path.exists(DB_PATH):
        print("⚠️ No database found. Nothing to verify.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 获取 DB 里所有记录过的文件名
    try:
        cursor.execute("SELECT DISTINCT filename FROM inference_logs")
        db_files = {row[0] for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        print("⚠️ Database table not ready yet.")
        return

    print(f"📋 Database contains records for {len(db_files)} files.")
    
    issues_found = 0
    fixed_count = 0

    # 2. 遍历检查每一个在 DB 里宣称“已完成”的文件
    for filename in tqdm(db_files, desc="Verifying Integrity"):
        file_path = os.path.join(INPUT_FOLDER, filename)
        
        # 如果文件在硬盘上不见了，跳过
        if not os.path.exists(file_path):
            continue

        # A. 从 DB 里重建该文件“应该”有的样子 (Source of Truth)
        # 按 index 排序提取所有标记为 KEEP 的句子
        cursor.execute("""
            SELECT sentence FROM inference_logs 
            WHERE filename = ? AND model_prediction = 'KEEP'
            ORDER BY sentence_index ASC
        """, (filename,))
        
        kept_sentences_db = [row[0] for row in cursor.fetchall()]
        expected_text = " ".join(kept_sentences_db)

        # B. 读取硬盘上文件“实际”的样子
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                actual_text = f.read()
        except Exception:
            continue

        # C. 比较 (简单比较长度或内容)
        # 注意：完全字符串相等比较可能因为空格处理略有不同而失败，
        # 所以我们这里做一个逻辑判断：如果硬盘上的文件包含了 DB 里标记为 REMOVE 的内容，肯定有问题。
        
        # 更简单的判断：如果硬盘文件长度 明显大于 DB 预期长度，说明没删干净
        # 或者直接：既然 DB 是真理，我们无条件信任 DB 的 KEEP 结果，直接重写一次文件即可。
        # 为了效率，我们只在长度差异大时重写，或者直接全部检查一遍。
        
        # 这里我们采用“字符级比对”策略来决定是否需要修复
        # 考虑到空格差异，我们去掉所有空格比较
        clean_expected = expected_text.replace(" ", "")
        clean_actual = actual_text.replace(" ", "")

        if clean_expected != clean_actual:
            issues_found += 1
            # D. 修复：用 DB 的数据覆盖文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(expected_text)
            fixed_count += 1
            # print(f"🔧 Fixed incomplete write for: {filename}")

    conn.close()
    print("\n" + "="*30)
    print("✅ Consistency Check Complete!")
    print(f"⚠️  Inconsistent files found: {issues_found}")
    print(f"🔧 Files repaired (overwritten with DB data): {fixed_count}")
    print("="*30)

if __name__ == "__main__":
    # 确保 NLTK 用于读取 (虽然这里主要靠 DB)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    verify_and_fix_consistency()