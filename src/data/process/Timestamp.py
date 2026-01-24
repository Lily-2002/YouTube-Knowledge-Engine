import difflib
import re

def assign_timestamps_to_chunks(data_json, chunks):
    """
    Inputs:
        data_json: The raw JSON list from YouTube (contains 'start', 'duration', 'text')
        chunks: A list of strings (e.g., ["Chunk 1 text...", "Chunk 2 text..."])
    
    Returns:
        A list of dicts: [{'text': "...", 'start': 10.5, 'end': 20.0}, ...]
    """

    # --- 1. FLATTEN RAW JSON (Source of Truth) ---
    raw_words_map = []
    # Adjust this path if your JSON structure is different (e.g. data_json['transcript'])
    # Based on your previous upload, it is data_json[0]['transcript']
    transcript_segments = data_json[0]['transcript'] 

    for segment in transcript_segments:
        words = segment['text'].split()
        if not words: continue
        
        # Linearly interpolate time across words
        word_duration = segment['duration'] / len(words)
        start_time = segment['start']
        
        for i, word in enumerate(words):
            w_start = start_time + (i * word_duration)
            w_end = w_start + word_duration
            w_norm = re.sub(r'[^\w\s]', '', word).lower() # Normalize
            
            if w_norm: 
                raw_words_map.append({
                    'word_norm': w_norm,
                    'start': w_start,
                    'end': w_end
                })

    # --- 2. FLATTEN CHUNKS (But remember chunk ID) ---
    clean_words_map = []
    chunk_boundaries = [] # Stores (start_index, end_index) for each chunk in the clean_words_map
    
    current_idx = 0
    
    for chunk_id, chunk_text in enumerate(chunks):
        words = chunk_text.split()
        chunk_start_idx = current_idx
        
        for word in words:
            w_norm = re.sub(r'[^\w\s]', '', word).lower()
            if w_norm:
                clean_words_map.append({
                    'word_norm': w_norm,
                    'chunk_id': chunk_id,
                    'global_index': current_idx
                })
                current_idx += 1
        
        chunk_end_idx = current_idx - 1 # The index of the last word in this chunk
        chunk_boundaries.append((chunk_start_idx, chunk_end_idx))

    # --- 3. ALIGNMENT (Global Sequence Match) ---
    raw_strings = [x['word_norm'] for x in raw_words_map]
    clean_strings = [x['word_norm'] for x in clean_words_map]

    matcher = difflib.SequenceMatcher(None, clean_strings, raw_strings)
    
    # Map: Clean_Global_Index -> Raw_Index
    clean_to_raw_index = {}
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i in range(i1, i2):
                offset = i - i1
                clean_to_raw_index[i] = j1 + offset

    # --- 4. ASSIGN TIMES TO CHUNKS ---
    final_chunks = []
    
    # We iterate through our stored boundaries to reconstruct chunk timings
    for i, (start_idx, end_idx) in enumerate(chunk_boundaries):
        chunk_text = chunks[i]
        
        # Default to 0 or previous end time if no match found (fallback)
        chunk_start = 0.0
        chunk_end = 0.0
        
        # 4a. Find Start Time: Look for first aligned word in this chunk
        # We iterate from the chunk's start word to its end word
        for k in range(start_idx, end_idx + 1):
            if k in clean_to_raw_index:
                raw_idx = clean_to_raw_index[k]
                chunk_start = raw_words_map[raw_idx]['start']
                break # Found the first match, stop
        
        # 4b. Find End Time: Look for last aligned word in this chunk
        # We iterate backwards from end word to start word
        for k in range(end_idx, start_idx - 1, -1):
            if k in clean_to_raw_index:
                raw_idx = clean_to_raw_index[k]
                chunk_end = raw_words_map[raw_idx]['end']
                break # Found the last match, stop

        # 4c. Safety Check: If chunk was purely "new" words (unlikely)
        if chunk_start == 0.0 and chunk_end == 0.0 and i > 0:
            # Inherit from previous chunk's end to avoid 0.0 gaps
            chunk_start = final_chunks[-1]['end']
            chunk_end = final_chunks[-1]['end']

        final_chunks.append({
            "text": chunk_text,
            "start": round(chunk_start, 2),
            "end": round(chunk_end, 2)
        })
    return final_chunks

# --- Example Usage ---
# import json
# with open('data.json') as f: data = json.load(f)
# my_chunks = ["This is the first chunk.", "This is the second chunk of text."]
# results = assign_timestamps_to_chunks(data, my_chunks)
# print(results)