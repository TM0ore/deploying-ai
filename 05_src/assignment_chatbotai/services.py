

import os
import json
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv("05_src/.secrets")
load_dotenv("05_src/.env")

# Shared configuration

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "pitchfork_reviews"
API_GATEWAY_KEY = os.getenv("API_GATEWAY_KEY")
GATEWAY_URL = "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1"

#  iTunes Search API (Service 1)

def search_itunes(query: str, entity: str = "album", limit: int = 5) -> str:
  
    url = "https://itunes.apple.com/search"
    params = {
        "term": query,
        "entity": entity,
        "limit": limit,
        "media": "music",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"iTunes API call failed: {str(e)}"

    results = data.get("results", [])

    if not results:
        return f"No iTunes results found for '{query}'."

    # Transforms raw JSON 
    lines = [f"iTunes results for '{query}':\n"]

    for i, item in enumerate(results, 1):
        if entity == "album":
            artist = item.get("artistName", "Unknown Artist")
            album  = item.get("collectionName", "Unknown Album")
            genre  = item.get("primaryGenreName", "Unknown Genre")
            year   = item.get("releaseDate", "")[:4] if item.get("releaseDate") else "Unknown"
            tracks = item.get("trackCount", "?")
            lines.append(
                f"{i}. {album} by {artist} "
                f"({year}) — {genre} — {tracks} tracks"
            )
        elif entity == "musicArtist":
            artist = item.get("artistName", "Unknown")
            genre  = item.get("primaryGenreName", "Unknown Genre")
            lines.append(f"{i}. {artist} — {genre}")
        elif entity == "song":
            track  = item.get("trackName", "Unknown Track")
            artist = item.get("artistName", "Unknown Artist")
            album  = item.get("collectionName", "Unknown Album")
            year   = item.get("releaseDate", "")[:4] if item.get("releaseDate") else "Unknown"
            lines.append(f"{i}. {track} by {artist} — from {album} ({year})")

    return "\n".join(lines)


# Search Pitchfork Reviews (Service 2)

def get_chroma_collection():
    """Load the persistent ChromaDB collection."""
    embedding_function = OpenAIEmbeddingFunction(
        api_key="any value",
        model_name="text-embedding-3-small",
        api_base=GATEWAY_URL,
        default_headers={"x-api-key": API_GATEWAY_KEY},
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    return collection


def search_pitchfork(query: str, n_results: int = 3) -> str:
   
    try:
        collection = get_chroma_collection()
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
        )
    except Exception as e:
        return f"Pitchfork search failed: {str(e)}"

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return f"No Pitchfork reviews found for '{query}'."

    lines = [f"Pitchfork reviews relevant to '{query}':\n"]

    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        title  = meta.get("title", "Unknown")
        artist = meta.get("artist", "Unknown")
        score  = meta.get("score", "N/A")
        genre  = meta.get("genre", "unknown")
        lines.append(
            f"{i}. {title} by {artist} "
            f"(Score: {score}/10, Genre: {genre})\n"
            f"   Review excerpt: {doc[:300]}..."
        )

    return "\n".join(lines)


# Mood-Based Recommendations (Function Calling with @tool) (Service 3)

CURATED_ALBUMS = [
    # Roots Reggae / Dancehall
    {"title": "Exodus",              "artist": "Bob Marley & The Wailers", "genre": "reggae",    "era": "70s",  "moods": ["uplifting", "spiritual", "revolutionary"]},
    {"title": "Burnin'",             "artist": "Bob Marley & The Wailers", "genre": "reggae",    "era": "70s",  "moods": ["rebellious", "soulful", "revolutionary"]},
    {"title": "Catch a Fire",        "artist": "Bob Marley & The Wailers", "genre": "reggae",    "era": "70s",  "moods": ["chill", "soulful", "uplifting"]},
    {"title": "Mr. Mention",         "artist": "Buju Banton",              "genre": "dancehall", "era": "90s",  "moods": ["energetic", "party", "bold"]},
    {"title": "Voice of Jamaica",    "artist": "Buju Banton",              "genre": "dancehall", "era": "90s",  "moods": ["bold", "rebellious", "energetic"]},
    {"title": "Til Shiloh",          "artist": "Buju Banton",              "genre": "reggae",    "era": "90s",  "moods": ["spiritual", "soulful", "reflective"]},
    # Hip Hop
    {"title": "Illmatic",            "artist": "Nas",                      "genre": "hip hop",   "era": "90s",  "moods": ["reflective", "gritty", "soulful"]},
    {"title": "Ready to Die",        "artist": "The Notorious B.I.G.",     "genre": "hip hop",   "era": "90s",  "moods": ["gritty", "bold", "energetic"]},
    {"title": "To Pimp a Butterfly", "artist": "Kendrick Lamar",           "genre": "hip hop",   "era": "2010s","moods": ["political", "soulful", "reflective"]},
    {"title": "DAMN.",               "artist": "Kendrick Lamar",           "genre": "hip hop",   "era": "2010s","moods": ["intense", "reflective", "bold"]},
    {"title": "My Beautiful Dark Twisted Fantasy", "artist": "Kanye West", "genre": "hip hop",   "era": "2010s","moods": ["intense", "grandiose", "bold"]},
    {"title": "The Miseducation of Lauryn Hill",   "artist": "Lauryn Hill","genre": "hip hop",   "era": "90s",  "moods": ["soulful", "uplifting", "reflective"]},
    # Jazz
    {"title": "Kind of Blue",        "artist": "Miles Davis",              "genre": "jazz",      "era": "50s",  "moods": ["chill", "reflective", "late night"]},
    {"title": "A Love Supreme",      "artist": "John Coltrane",            "genre": "jazz",      "era": "60s",  "moods": ["spiritual", "intense", "late night"]},
    {"title": "Head Hunters",        "artist": "Herbie Hancock",           "genre": "jazz",      "era": "70s",  "moods": ["funky", "energetic", "chill"]},
    # Soul / R&B
    {"title": "What's Going On",     "artist": "Marvin Gaye",              "genre": "soul",      "era": "70s",  "moods": ["political", "soulful", "reflective"]},
    {"title": "Songs in the Key of Life", "artist": "Stevie Wonder",       "genre": "soul",      "era": "70s",  "moods": ["uplifting", "soulful", "joyful"]},
    {"title": "I Never Loved a Man the Way I Love You", "artist": "Aretha Franklin", "genre": "soul", "era": "60s", "moods": ["soulful", "powerful", "emotional"]},
    # Indie / Alternative
    {"title": "OK Computer",         "artist": "Radiohead",                "genre": "indie",     "era": "90s",  "moods": ["intense", "reflective", "melancholic"]},
    {"title": "In Rainbows",         "artist": "Radiohead",                "genre": "indie",     "era": "2000s","moods": ["melancholic", "chill", "late night"]},
    {"title": "Funeral",             "artist": "Arcade Fire",              "genre": "indie",     "era": "2000s","moods": ["emotional", "uplifting", "intense"]},
    {"title": "Blonde",              "artist": "Frank Ocean",              "genre": "r&b",       "era": "2010s","moods": ["melancholic", "reflective", "late night"]},
    {"title": "Channel ORANGE",      "artist": "Frank Ocean",              "genre": "r&b",       "era": "2010s","moods": ["chill", "soulful", "late night"]},
    # Electronic
    {"title": "Random Access Memories", "artist": "Daft Punk",            "genre": "electronic","era": "2010s","moods": ["party", "funky", "energetic"]},
    {"title": "Homework",            "artist": "Daft Punk",                "genre": "electronic","era": "90s",  "moods": ["party", "energetic", "funky"]},
]


@tool
def get_mood_recommendations(mood: str, genre: str = "", era: str = "") -> str:
    """
    Recommends albums from a list based on mood, genre, and era.
    Use this when the user asks for music recommendations based on how they
    feel, what they want to do, or what kind of vibe they are looking for.
    """
    mood_lower  = mood.lower().strip()
    genre_lower = genre.lower().strip()
    era_lower   = era.lower().strip()

    matches = []

    for album in CURATED_ALBUMS:
        mood_match  = any(mood_lower in m for m in album["moods"])
        genre_match = (not genre_lower) or (genre_lower in album["genre"].lower())
        era_match   = (not era_lower) or (era_lower == album["era"])

        if mood_match and genre_match and era_match:
            matches.append(album)

    if not matches:
        matches = [a for a in CURATED_ALBUMS if any(mood_lower in m for m in a["moods"])]

    if not matches:
        return f"No recommendations found for mood='{mood}', genre='{genre}', era='{era}'."

    lines = [f"Album recommendations for mood='{mood}'"
             + (f", genre='{genre}'" if genre else "")
             + (f", era='{era}'" if era else "")
             + ":\n"]

    for i, album in enumerate(matches[:5], 1):
        lines.append(
            f"{i}. {album['title']} by {album['artist']} "
            f"({album['era']}, {album['genre']})"
        )

    return "\n".join(lines)