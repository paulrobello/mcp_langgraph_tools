import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console
from mcp.types import InitializeResult
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition

from .mcp_tool_node import mcp_tool_list, McpToolNode

load_dotenv()

console = Console()

# Define MCP Server Parameters
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-brave-search"],
    env={
        "BRAVE_API_KEY": os.environ.get("BRAVE_API_KEY"),  # get a free key from BRAVE
        "PATH": os.environ.get("PATH"),  # adding PATH helps MCP spawned process find things your path
    },
)

# Works with any tool capable LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
# llm = ChatOpenAI(model="gpt-4o")
# llm = ChatOllama(model="llama3.2:latest")


async def amain():
    """Async main function to connect to MCP."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            res: InitializeResult = await session.initialize()
            try:
                llm_tools = await mcp_tool_list(session)
                console.print("MCP Tools:", llm_tools)
            except Exception as _:
                llm_tools = []
                console.print("MCP Server reports no tools available.")

            llm_with_tools = llm.bind_tools(llm_tools)
            sys_msg = SystemMessage(content="You are a helpful assistant. Use available tools to assist the user.")

            # Graph Node
            def assistant(state: MessagesState):
                return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

            # Build graph
            builder = StateGraph(MessagesState)
            # Define nodes: these do the work
            builder.add_node("assistant", assistant)
            builder.add_node("tools", await McpToolNode(session, handle_tool_errors=True).init_funcs())

            # Define edges: these determine how the control flow moves
            builder.add_edge(START, "assistant")
            builder.add_conditional_edges(
                "assistant",
                # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
                # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
                tools_condition,
            )
            builder.add_edge("tools", "assistant")
            react_graph = builder.compile()

            messages = [HumanMessage(content="Search for Paul Robello the Principal Solution Architect")]
            # Invoke the graph with initial messages
            messages = await react_graph.ainvoke({"messages": messages})
            for m in messages["messages"]:
                m.pretty_print()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
