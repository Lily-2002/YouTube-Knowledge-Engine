import json
from pathlib import Path
from datasketch import MinHash, MinHashLSH

def clean_duplicates_in_folder(folder_path, threshold=0.8):
    # Initialize LSH with Jaccard similarity threshold (0.8 = 80% similar)
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    
    # Cache to store file contents: { file_path_object : [list_of_units] }
    files_cache = {}
    
    # Flat list to iterate through all units linearly
    all_units_ref = []

    # --- Step 1: Read, Generate IDs, and Load Cache ---
    path_obj = Path(folder_path)
    for file_path in path_obj.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list): continue

            for index, unit in enumerate(data):
                # Generate ID: filename (stem) + index (1-based)
                # Example: "lecture_notes_1"
                unit['id'] = f"{file_path.stem}_{index + 1}"
                
                # Store reference for processing
                all_units_ref.append(unit)

            files_cache[file_path] = data
            
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    print(f"Loaded {len(all_units_ref)} units. Starting deduplication...")

    # --- Step 2: Identify Duplicates with MinHash ---
    ids_to_delete = set()
    
    for unit in all_units_ref:
        text = unit.get('text', "")
        
        # Skip empty text units to avoid errors
        if not text: continue
        
        # Create MinHash fingerprint
        m = MinHash(num_perm=128)
        # Simple whitespace tokenization
        for token in text.split():
            m.update(token.encode('utf8'))
            
        # Query LSH to see if this fingerprint exists already
        duplicates = lsh.query(m)
        
        if duplicates:
            # If query returns result, this unit is a duplicate of a previous one.
            # We mark THIS unit for deletion.
            ids_to_delete.add(unit['id'])
        else:
            # If no result, this is a unique/new unit. Index it.
            lsh.insert(unit['id'], m)

    print(f"Found {len(ids_to_delete)} duplicates to remove.")

    # --- Step 3: Delete and Save Changes ---
    for file_path, units in files_cache.items():
        original_count = len(units)
        
        # Filter: Keep unit ONLY if its ID is NOT in the delete set
        cleaned_units = [u for u in units if u['id'] not in ids_to_delete]
        
        # Only write to disk if deletions actually occurred
        if len(cleaned_units) < original_count:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_units, f, indent=4, ensure_ascii=False)
            print(f"Updated {file_path.name}: Removed {original_count - len(cleaned_units)} duplicates.")

if __name__ == "__main__":
    clean_duplicates_in_folder("./data/Timestamp")