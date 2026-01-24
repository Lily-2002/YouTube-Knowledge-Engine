import re

def re_punc(model, text):
    # 1. Manually chunk the text to avoid the "chunk size too large" Assertion Error
    # A safe limit for this library is around 10k-15k characters per pass.
    SAFE_CHUNK_SIZE = 10000 
    
    processed_chunks = []
    start_index = 0
    text_len = len(text)
    
    while start_index < text_len:
        # Calculate tentative end index
        end_index = min(start_index + SAFE_CHUNK_SIZE, text_len)
        
        # If we are not at the very end, back up to the nearest space
        # This prevents cutting a word in half (e.g. "com-puter")
        if end_index < text_len:
            last_space = text.rfind(' ', start_index, end_index)
            if last_space != -1:
                end_index = last_space
        
        # Get the slice
        current_chunk = text[start_index:end_index]
        
        # Process this chunk using the model
        if current_chunk.strip():
            try:
                # The model can handle 10k chars safely
                punctuated_chunk = model.restore_punctuation(current_chunk)
                processed_chunks.append(punctuated_chunk)
            except Exception as e:
                print(f"⚠️ Warning: Chunk failed (idx {start_index}). Using raw text. Error: {e}")
                processed_chunks.append(current_chunk)
        
        # Move start_index forward. 
        # If we split at a space, skip the space (+1).
        start_index = end_index + 1 if end_index < text_len else end_index

    # 2. Join the punctuated chunks back together
    # using " " ensures we don't merge words accidentally
    full_result = " ".join(processed_chunks)

    # 3. Apply your requested Regex for capitalization
    # This capitalizes the first letter after . ! ?
    return re.sub(
        r"(^|[.!?]\s+)([a-z])", 
        lambda m: m.group(1) + m.group(2).upper(), 
        full_result
    )