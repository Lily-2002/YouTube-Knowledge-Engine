import json
import re
def clean_filler_words(text):
    fillers = ["um", "uh", "hmm", "ah", "er", "uh-huh"]
    pattern = r'\b(' + '|'.join(fillers) + r')\b'
    clean_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text
def merge_transcripts_to_string(data):
    try:
        # with open(file_path, "r", encoding="utf-8") as f:
        #     data = json.load(f)
        
        all_text_segments = []

        # Iterate through each video entry in the list
        for video_entry in data:
            # Iterate through the list of transcript dictionaries
            for snippet in video_entry.get("transcript", []):
                # Extract the 'text' value
                text = snippet.get("text", "")
                if text:
                    all_text_segments.append(text)

        # Merge all segments with a space
        full_string = " ".join(" ".join(all_text_segments).split())
        
        return clean_filler_words(full_string)

    except Exception as e:
        return f"Error: {e}"


# merged_text = merge_transcripts_to_string("/home/fulian/RAG/data/raw/data.json")

