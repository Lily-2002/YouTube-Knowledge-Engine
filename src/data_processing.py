from data import merge_transcripts_to_string
from data import re_punc
from data import fetch_all_transcripts_with_metadata
from deepmultilingualpunctuation import PunctuationModel
from data import assign_timestamps_to_chunks
from data import get_top_cs_videos
import json
import asyncio
# Get raw data
video_id = "sVcwVQRHIc8"
data= asyncio.run(fetch_all_transcripts_with_metadata([video_id]))
json_ready_data = [item.model_dump() for item in data]
# merge
merged_text = merge_transcripts_to_string(json_ready_data)
model = PunctuationModel("/home/fulian/RAG/fullstop-punctuation-multilang-sonar-base")
result = re_punc(model,str(merged_text))
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
texts = text_splitter.split_text(result)
r = assign_timestamps_to_chunks(json_ready_data,texts)
with open(f'/home/fulian/RAG/data/processed/{video_id}.json', 'w', encoding='utf-8') as f:
        json.dump(r, f, indent=4, ensure_ascii=False)