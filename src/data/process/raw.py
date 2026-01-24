from concurrent.futures import ThreadPoolExecutor
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, VideoUnavailable, TranscriptsDisabled, IpBlocked
from tenacity import retry, wait_fixed, retry_if_exception_type, stop_after_attempt
from typing import Literal
import asyncio
import httpx
from pydantic import BaseModel
from fake_useragent import UserAgent
import random
import json

ACCEPT_LANGUAGES = ["en-US,en;q=0.9"]
REFERERS = ["https://www.youtube.com/"]
TRANSCRIPT_FETCH_TIMEOUT = 100

def get_realistic_headers() -> dict:
    ua = UserAgent(platforms='desktop')
    return {
        "User-Agent": ua.random,
        "Accept": "text/html,application/xhtml+xml,...",
        "Accept-Language": random.choice(ACCEPT_LANGUAGES),
        "Referer": random.choice(REFERERS),
        # Additional modern headers can go here
    }
class FetchAndMetaResponse(BaseModel):
    video_id: str
    transcript: list[dict]
headers = get_realistic_headers()
httpx_client = httpx.Client(timeout=TRANSCRIPT_FETCH_TIMEOUT, headers=headers)

# Global API and thread pool
executor = ThreadPoolExecutor(max_workers=30)

@retry(
    retry=retry_if_exception_type(IpBlocked),
    wait=wait_fixed(1),
    stop=stop_after_attempt(2)
)
def fetch_transcript_with_snippet(video_id: str) -> dict | None:
    try:
        ytt_api = YouTubeTranscriptApi(http_client=httpx_client)
        transcript = ytt_api.fetch(video_id).to_raw_data()

        return {
            "video_id": video_id,
            "transcript": transcript,
        }
    except (NoTranscriptFound, VideoUnavailable, TranscriptsDisabled):
        return None
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return None
video_ids = ['sVcwVQRHIc8']
async def fetch_all_transcripts_with_metadata(video_ids: list[str]) -> list[FetchAndMetaResponse]:
    async def run_in_thread(vid):
        return await asyncio.to_thread(fetch_transcript_with_snippet, vid)

    tasks = [run_in_thread(v) for v in video_ids]
    results = await asyncio.gather(*tasks)

    return [
        FetchAndMetaResponse(
            video_id=result["video_id"],
            transcript=result["transcript"],
        )
        for result in results if result
    ]
# data= asyncio.run(fetch_all_transcripts_with_metadata(video_ids))

# for item in data:
#         print(item.model_dump_json(indent=2))
# json_ready_data = [item.model_dump() for item in data]
# with open('/home/fulian/RAG/data/raw/data.json', 'w') as f:
#     json.dump(json_ready_data, f, indent=4)