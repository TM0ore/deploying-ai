from typing import Literal
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict, Annotated

from dotenv import load_dotenv

from utils.logger import get_logger


_logs = get_logger(__name__)

load_dotenv(".env")
load_dotenv(".secrets")


model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0.7
)


client = MultiServerMCPClient(
    {
        "music_server": {
            # make sure you start your music server on port 8000
            "url": "http://sciuroid-jackeline-overventurously.ngrok-free.dev/mcp",
            "transport": "streamable_http",
        }
    }
)
async def get_tools_from_mcp_client(client: MultiServerMCPClient):
    """Fetch tools from MCP client"""
    tools = await client.get_tools()
    return tools

dev_prompt = """
    You are a helpful assistant tasked with offering music recommendations. 
    Use the music_server when a user asks for a recommendation. 
    Respond with the tool's output directly.
"""


chat_agent = init_chat_model(
    "openai:gpt-4o-mini"
)

# @traceable(run_type="llm")
async def call_model(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    tools = await get_tools_from_mcp_client(client)
    response = chat_agent.bind_tools(tools).invoke(state["messages"])
    return {
        "messages": [response]
    }


async def get_graph():
    tools = await get_tools_from_mcp_client(client)
    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    return graph

async def run_graph():
    graph = await get_graph()
    response =  await graph.ainvoke({'messages': HumanMessage(content="what is a good album?")})
    return response['messages']

if __name__ == "__main__":
    import asyncio
    result = asyncio.run(run_graph())
    print(result)