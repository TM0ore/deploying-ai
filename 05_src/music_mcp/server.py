from fastmcp import FastMCP
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pydantic import BaseModel, Field

import sqlalchemy as sa
import pandas as pd

from dotenv import load_dotenv
import ngrok
import os

from utils.logger import get_logger

# Load environment variables and secrets
load_dotenv()
load_dotenv(".secrets")

# Setup Logger
_logs = get_logger(__name__)


MCP_DOMAIN = os.getenv("MCP_DOMAIN")

vector_db_client_url="http://localhost:8000"
chroma = chromadb.HttpClient(host=vector_db_client_url)
collection = chroma.get_collection(name="pitchfork_reviews", 
                                   embedding_function=OpenAIEmbeddingFunction(
                                       api_key = os.getenv("OPENAI_API_KEY"),
                                       model_name="text-embedding-3-small")
                                   )

# Initialize MCP Server
mcp = FastMCP(
    name="music_server",
    instructions="""
    This server provides music recommendations based on Pitchfork reviews.
    """
)

class MusicReviewData(BaseModel):
    """Structured music review data response."""
    title: str = Field(..., description="The title of the album.")
    artist: str = Field(..., description="The artist of the album.")
    review: str = Field(..., description="The review of the album.")
    year: int = Field(None, description="The release year of the album.")
    score: float = Field(None, description="The Pitchfork score of the album. An album with a score of 6.5 and above is considered good. A score of 8.0 and above indicates a great album.")


@mcp.tool
def music_review_service(query: str, n_results: int = 1) -> str:
    """Fetches music review data based on the query. Returns n_results reviews."""
    response = generate_response(query, collection, n_results)
    return response


def additional_details(review_id:str):
    engine = sa.create_engine(os.getenv("SQL_URL"))
    query = f"""
    SELECT r.reviewid,
		r.title,
		r.artist,
		r.score,
		g.genre
    FROM reviews AS r
    LEFT JOIN genres as g
	    ON r.reviewid = g.reviewid
    WHERE r.reviewid = '{review_id}'
    """
    with engine.connect() as conn:
        result = pd.read_sql(query, conn)
    if not result.empty:
        row = result.iloc[0]
        details = {
            "reviewid": row['reviewid'],
            "album": row['title'],
            "score": row['score'],
            "artist": row['artist']
        }
        return details
    else:
        return {}
    
def get_reviewid_from_custom_id(custom_id:str):
    return custom_id.split('_')[0]

def get_context_data(query:str, collection:chromadb.api.models.Collection, top_n:int):
    results = collection.query(
        query_texts=[query],
        n_results=top_n
    )
    context_data = []
    for idx, custom_id in enumerate(results['ids'][0]):
        review_id = get_reviewid_from_custom_id(custom_id)
        details = additional_details(review_id)
        details['text'] = results['documents'][0][idx]
        context_data.append(details)
    return context_data

def generate_prompt(query:str, collection:chromadb.api.models.Collection, top_n:int):
    context_data = get_context_data(query, collection, top_n)
    prompt = f"Given a query, provide a detailed response using the context from relevant Pitchfork reviews. The context will contain references to {top_n} album reviews.\n\n"
    prompt += f"The score is numeric and its scale is from 0 to 10, with 10 being the highest rating. Any album with a score greater than 8.0 is considered a must-listen; album with a score greater than 6.5 is good.\n\n"
    prompt += f"<query>{query}</query>\n\n"
    prompt += "<context>\n"
    for k, context in enumerate(context_data):
        prompt += f"<album {k}>\n"
        prompt += f"- Album Title: {context.get('album', 'N/A')}\n" 
        prompt += f"- Album Artist: {context.get('artist', 'N/A')}\n"
        prompt += f"- Album Score: {context.get('score', 'N/A')}\n"
        prompt += f"- Review Quote: {context.get('text', 'N/A')}\n"
        prompt += f"</album {k}>\n\n"
    prompt += "</context>\n\n"
    prompt += "\nBased on the context and nothing else, provide a detailed response to the query."
    return prompt

def generate_response(query:str, collection:chromadb.api.models.Collection, top_n:int=1):
    prompt = generate_prompt(query, collection, top_n)
    print("Generated Prompt:\n", prompt)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides information based on Pitchfork reviews."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    _logs.debug(f'Music review response: {response.choices[0].message.content}')
    return response.choices[0].message.content

if __name__ == "__main__":
    listener = ngrok.forward("localhost:3000", authtoken_from_env=True,
                                domain=MCP_DOMAIN)
    _logs.info(f'Ngrok tunnel established at {listener.url()}')
    mcp.run(
        transport="http",
        host="localhost", 
        port=3000, 
    )
