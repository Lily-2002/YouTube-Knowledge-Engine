import pickle
import os
import re
from googleapiclient.discovery import build
from collections import defaultdict

# --- CONFIGURATION ---
UNIVERSITY_CHANNELS = {
    "MIT OpenCourseWare": "UCEBb1b_L6zDS3xTUrIALZOw",
    "Stanford Online": "UCBa5G_ESCn8Yd4vw5U-gIcg",
    "freecamp":"UC8butISFwT-Wl7EV0hUK0BQ"
    # Add UC Berkeley or others here if you find their specific "Courses" channel ID
}

# Minimum seconds to be considered a "Lecture" (Skips Shorts/Intros)
MIN_DURATION_SECONDS = 300 

def parse_duration(duration):
    """Converts ISO 8601 duration (PT#M#S) to total seconds."""
    if not duration: return 0
    
    hours = re.search(r'(\d+)H', duration)
    minutes = re.search(r'(\d+)M', duration)
    seconds = re.search(r'(\d+)S', duration)
    
    total_seconds = 0
    total_seconds += int(hours.group(1)) * 3600 if hours else 0
    total_seconds += int(minutes.group(1)) * 60 if minutes else 0
    total_seconds += int(seconds.group(1)) if seconds else 0
    return total_seconds

def normalize_course_title(title):
    """
    Normalizes titles to find duplicates.
    Ex: "CS50 - 2021" and "CS50 2022" become "cs50"
    """
    t = title.lower()
    # Remove years (2010-2029)
    t = re.sub(r'\b20[12][0-9]\b', '', t)
    # Remove semesters
    t = re.sub(r'\b(fall|spring|summer|winter)\b', '', t)
    # Remove special chars
    t = re.sub(r'[\-\|:\[\]\(\)]', '', t)
    return " ".join(t.split())

def get_channel_playlists(youtube, channel_id):
    """Fetches ALL playlists from a channel."""
    playlists = []
    next_page_token = None
    
    while True:
        try:
            request = youtube.playlists().list(
                part="snippet",
                channelId=channel_id,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response.get('items', []):
                playlists.append({
                    'id': item['id'],
                    'title': item['snippet']['title'],
                    'publishedAt': item['snippet']['publishedAt']
                })
                
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        except Exception as e:
            print(f"   ⚠️ Error fetching playlists: {e}")
            break
            
    return playlists

def filter_newest_courses(playlists):
    """
    Groups playlists by normalized title and keeps only the newest version.
    """
    grouped = defaultdict(list)
    
    for p in playlists:
        norm_title = normalize_course_title(p['title'])
        # Skip very short titles or generic names often used for junk
        if len(norm_title) < 4: continue 
        grouped[norm_title].append(p)
        
    selected_playlists = []
    dedup_count = 0
    
    for course_name, versions in grouped.items():
        # Sort by published date descending (newest first)
        versions.sort(key=lambda x: x['publishedAt'], reverse=True)
        selected_playlists.append(versions[0])
        if len(versions) > 1:
            dedup_count += (len(versions) - 1)
            
    print(f"   ✂️ Removed {dedup_count} older duplicate course versions.")
    return selected_playlists

def get_videos_from_playlist_filtered(youtube, playlist_id):
    """
    Fetches video IDs from a playlist AND filters by duration (Method 1).
    """
    valid_video_ids = []
    next_page_token = None
    
    while True:
        try:
            # Step 1: Get Video IDs from Playlist
            pl_request = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )
            pl_response = pl_request.execute()
            
            batch_ids = []
            for item in pl_response.get('items', []):
                vid_id = item['contentDetails']['videoId']
                batch_ids.append(vid_id)
            
            if not batch_ids:
                break
            
            # Step 2: Get Duration for filtering (The Shorts Blocker)
            vid_request = youtube.videos().list(
                part="contentDetails",
                id=','.join(batch_ids)
            )
            vid_response = vid_request.execute()
            
            for item in vid_response.get('items', []):
                duration_str = item['contentDetails'].get('duration')
                
                # Skip live streams or invalid durations
                if not duration_str: continue
                
                # CHECK: Is it longer than 3 minutes?
                if parse_duration(duration_str) >= MIN_DURATION_SECONDS:
                    valid_video_ids.append(item['id'])

            next_page_token = pl_response.get('nextPageToken')
            if not next_page_token:
                break
                
        except Exception as e:
            print(f"   ⚠️ Error processing playlist items: {e}")
            break
            
    return valid_video_ids

def get_university_courses(api_key='AIzaSyBHcVp7fOIc1oeT-aKasAYKSa4_PzbDYUY'):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    pkl_filename = "university_course_ids.pkl"
    
    # Load Progress
    if os.path.exists(pkl_filename):
        with open(pkl_filename, "rb") as f:
            all_video_ids = pickle.load(f)
        print(f"🔄 Resuming: Loaded {len(all_video_ids)} IDs.")
    else:
        all_video_ids = []

    seen_ids = set(all_video_ids)
    
    # We also track processed playlists to avoid re-scanning fully completed courses
    # (Simple logic: if we have 50k videos, we assume we did the work, but for safety we re-scan deduplication)

    for uni_name, channel_id in UNIVERSITY_CHANNELS.items():
        print(f"\n🏛️ --- Processing Channel: {uni_name} ---")
        
        # 1. Fetch
        raw_playlists = get_channel_playlists(youtube, channel_id)
        print(f"   Found {len(raw_playlists)} raw playlists.")
        
        # 2. Filter
        courses = filter_newest_courses(raw_playlists)
        print(f"   Processing {len(courses)} unique courses...")
        
        # 3. Extract Videos
        for i, course in enumerate(courses):
            # Optional: Simple log to show progress
            print(f"   [{i+1}/{len(courses)}] Scanning: {course['title']}")
            
            new_ids = get_videos_from_playlist_filtered(youtube, course['id'])
            
            added_count = 0
            for vid in new_ids:
                if vid not in seen_ids:
                    all_video_ids.append(vid)
                    seen_ids.add(vid)
                    added_count += 1
            
            # Save every 5 playlists to be safe
            if i % 5 == 0 and added_count > 0:
                with open(pkl_filename, "wb") as f:
                    pickle.dump(all_video_ids, f)
                print(f"   💾 Saved Progress. Total IDs: {len(all_video_ids)}")

    # Final Save
    with open(pkl_filename, "wb") as f:
        pickle.dump(all_video_ids, f)
        
    print(f"\n✅ DONE! Collected a total of {len(all_video_ids)} video IDs.")
    return all_video_ids