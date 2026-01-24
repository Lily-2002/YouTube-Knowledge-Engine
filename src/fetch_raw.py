from data import merge_transcripts_to_string
from data import re_punc
from data import fetch_all_transcripts_with_metadata
from deepmultilingualpunctuation import PunctuationModel
from data import assign_timestamps_to_chunks
from data import get_university_courses
import json
import asyncio
import pickle
import pickle
import os
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)

def process_json_files(directory_path = "/home/fulian/RAG/data/raw"):
    base_path = Path(directory_path)
    all_data = {}
    for file_path in base_path.glob("*.json"):
        file_name_only = file_path.stem
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data[file_name_only] = data
            print(f"Successfully read: {file_name_only}")
        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")

    return all_data
def extract_ids_from_pickle(pkl_filename="university_course_ids.pkl"):
    # Check if the file exists to avoid errors
    if not os.path.exists(pkl_filename):
        print(f"Error: The file '{pkl_filename}' was not found.")
        return []

    try:
        with open(pkl_filename, "rb") as f:
            video_ids = pickle.load(f)
        
        print(f"Successfully extracted {len(video_ids)} video IDs.")
        return video_ids
    
    except Exception as e:
        print(f"An error occurred while reading the pickle: {e}")
        return []
def get_raw_data():
    datas = process_json_files()
    ids = datas.keys()
    for id in ids:
        id = str(id)
        # print(id)
        # data= asyncio.run(fetch_all_transcripts_with_metadata([id]))
        # if not data:
        #     print(f"⚠️ Skip: No transcript found for {id}")
        #     continue
        # json_ready_data = [item.model_dump() for item in data]
        # with open(f"/home/fulian/RAG/data/raw/{id}.json", "w") as f:
        #     json.dump(json_ready_data, f, indent=4)
        # merge
        json_ready_data = datas[id]
        merged_text = merge_transcripts_to_string(json_ready_data)
        model = PunctuationModel("/home/fulian/RAG/fullstop-punctuation-multilang-sonar-base")
        result = re_punc(model,str(merged_text))
        with open(f"/home/fulian/RAG/data/processed/punc_{id}.txt", "w", encoding="utf-8") as f:
            f.write(result)
        texts = text_splitter.split_text(result)
        r = assign_timestamps_to_chunks(json_ready_data,texts)
        with open(f'/home/fulian/RAG/data/Timestamp/{id}.json', 'w', encoding='utf-8') as f:
            json.dump(r, f, indent=4, ensure_ascii=False)
if __name__ == '__main__':
    # google_api = "AIzaSyAoL_jcQYHu0MfRB1t94NkbSqJX6krNNkg"
    # video_ids = get_university_courses()
    # video_ids = extract_ids_from_pickle()
    # get_raw_data(video_ids)
    get_raw_data()
