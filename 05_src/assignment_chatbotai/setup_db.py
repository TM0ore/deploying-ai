import os
import json

from dotenv import load_dotenv
load_dotenv("05_src/.secrets")
load_dotenv("05_src/.env")

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Paths
DOCUMENTS_DIR = "05_src/documents"
CHROMA_DIR = "05_src/assignment_chatbotai/chroma_db"
COLLECTION_NAME = "pitchfork_reviews"


def load_jsonl(filepath: str) -> list[dict]:
    """Load a JSON lines file and return a list of dicts."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# Load source data
print("Loading Pitchfork data files...")

content = load_jsonl(os.path.join(DOCUMENTS_DIR, "pitchfork_content_small.jsonl"))
reviews = load_jsonl(os.path.join(DOCUMENTS_DIR, "pitchfork_reviews.jsonl"))
genres  = load_jsonl(os.path.join(DOCUMENTS_DIR, "pitchfork_genres.jsonl"))

print(f"  Content records : {len(content)}")
print(f"  Review records  : {len(reviews)}")
print(f"  Genre records   : {len(genres)}")

# Build lookup dictionaries
review_lookup = {str(r["reviewid"]): r for r in reviews}

genre_lookup = {}
for g in genres:
    rid = str(g["reviewid"])
    if rid not in genre_lookup:
        genre_lookup[rid] = g.get("genre", "unknown")

# Prepare documents
print("\nPreparing documents...")

documents = []
metadatas = []
ids = []
seen_ids = set()

for item in content:
    reviewid = str(item.get("reviewid", ""))
    text = item.get("content", "")

    # Skip empty content
    if not text or not text.strip():
        continue

    # Skip duplicate IDs
    doc_id = f"review_{reviewid}"
    if doc_id in seen_ids:
        continue
    seen_ids.add(doc_id)

    # Get metadata
    review = review_lookup.get(reviewid, {})
    title  = review.get("title", "Unknown")
    artist = review.get("artist", "Unknown")
    score  = review.get("score", 0.0)
    genre  = genre_lookup.get(reviewid, "unknown")

    # Truncate to 1000 characters
    documents.append(text[:1000].strip())
    metadatas.append({
        "reviewid": reviewid,
        "title"   : str(title),
        "artist"  : str(artist),
        "score"   : float(score) if score else 0.0,
        "genre"   : str(genre),
    })
    ids.append(doc_id)

print(f"  Documents prepared: {len(documents)}")

# Set up ChromaDB with file persistence
print(f"\nSetting up ChromaDB at: {CHROMA_DIR}")
os.makedirs(CHROMA_DIR, exist_ok=True)

# Embedding function using the course API gateway
embedding_function = OpenAIEmbeddingFunction(
    api_key="any value",
    model_name="text-embedding-3-small",
    api_base="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
)

# Create persistent client and collection
client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
)

print(f"  Collection created: {COLLECTION_NAME}")

# Deduplicate by ID before adding
seen = {}
for doc, meta, id_ in zip(documents, metadatas, ids):
    if id_ not in seen:
        seen[id_] = (doc, meta)

final_ids   = list(seen.keys())
final_docs  = [seen[i][0] for i in final_ids]
final_metas = [seen[i][1] for i in final_ids]

# Add documents in batches
BATCH_SIZE = 500
total = len(final_ids)
print(f"\nAdding {total} documents in batches of {BATCH_SIZE}...")

for i in range(0, total, BATCH_SIZE):
    collection.add(
        documents=final_docs[i : i + BATCH_SIZE],
        metadatas=final_metas[i : i + BATCH_SIZE],
        ids=final_ids[i : i + BATCH_SIZE],
    )
    end = min(i + BATCH_SIZE, total)
    print(f"  Added {end}/{total} documents")

print(f"\nDone. Collection '{COLLECTION_NAME}' ready at {CHROMA_DIR}")
print(f"Total documents in collection: {collection.count()}")