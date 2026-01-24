import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

# --- CONFIG ---
CSV_PATH = '/home/fulian/RAG/jobs_10k_cleaned copy.csv'
MODEL_PATH = "/home/fulian/RAG/bge-m3"
COLLECTION_NAME = "linkedin_jobs"
UPLOAD_BATCH_SIZE = 100   # How many to send to Qdrant at once
ENCODE_BATCH_SIZE = 12    # Small batch for GPU safety
# --------------

print("1. Connecting to Qdrant...")
client = QdrantClient(url="http://localhost:6333")

# --- STEP 1: CHECK EXISTING COLLECTION ---
try:
    collection_info = client.get_collection(COLLECTION_NAME)
    # We assume sequential IDs (0, 1, 2...). The count tells us where to resume.
    existing_count = collection_info.points_count
    print(f"   Found existing collection '{COLLECTION_NAME}' with {existing_count} vectors.")
except Exception:
    print(f"   Collection '{COLLECTION_NAME}' not found. Creating new...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.DOT),
    )
    existing_count = 0

print("2. Loading Data...")
df = pd.read_csv(CSV_PATH)

total_rows = len(df)
print(f"   Total Rows in CSV: {total_rows}")

# --- STEP 2: CALCULATE NEW DATA ---
if existing_count >= total_rows:
    print("\n[INFO] Collection is up to date. No new rows to process.")
    exit()

rows_to_process = total_rows - existing_count
print(f"   New Rows to Add: {rows_to_process} (Skipping first {existing_count})")

print("3. Loading Model...")
model = SentenceTransformer(MODEL_PATH, device="cuda")

print("4. Starting Incremental Processing...")

# --- STEP 3: LOOP STARTING FROM EXISTING_COUNT ---
# We start the range at 'existing_count' instead of 0
for start_idx in tqdm(range(existing_count, total_rows, UPLOAD_BATCH_SIZE)):
    end_idx = min(start_idx + UPLOAD_BATCH_SIZE, total_rows)
    
    # Get the chunk of data
    batch_df = df.iloc[start_idx:end_idx]
    
    # Extract lists
    batch_sentences = batch_df['description'].tolist()
    batch_titles = batch_df['title'].tolist()
    batch_ids = batch_df['id'].tolist()
    batch_companies = batch_df['company'].tolist()
    batch_locations = batch_df['location'].tolist()
    batch_date = batch_df['date_posted'].tolist()
    batch_type = batch_df['job_type'].tolist()
    # GENERATE EMBEDDINGS
    batch_embeddings = model.encode(
        batch_sentences, 
        batch_size=ENCODE_BATCH_SIZE, 
        show_progress_bar=False,
        normalize_embeddings=True
    )
    
    # PREPARE POINTS
    points = []
    for i in range(len(batch_embeddings)):
        # Calculate the absolute index (ID) for this point
        current_absolute_id = start_idx + i
        
        payload = {
            "title": str(batch_titles[i]),
            "job_id": str(batch_ids[i]),
            "original_index": current_absolute_id,
            "company": str(batch_companies[i]),
            "location": str(batch_locations[i]),
            "date_posted": str(batch_date[i]),
            "job_type": str(batch_type[i])
        }
        
        points.append(PointStruct(
            id=current_absolute_id,  # Keeps IDs sequential (5000, 5001, etc.)
            vector=batch_embeddings[i].tolist(),
            payload=payload
        ))

    # UPLOAD TO QDRANT
    client.upsert(
        collection_name=COLLECTION_NAME,
        wait=False,
        points=points
    )

print(f"\nDone! Added {rows_to_process} new vectors to Qdrant.")