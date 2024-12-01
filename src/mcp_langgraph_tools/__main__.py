import asyncio
from typing import Any

import orjson as json
import os
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client, get_default_environment
from pydantic import BaseModel, Field
from rich.console import Console
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition

from .mcp_tool_node import mcp_tool_list, mcp_tool_node, combined_tool_node, McpToolNode

load_dotenv()

console = Console()

config_file = Path("config.json")


class McpServerConfig(StdioServerParameters):
    model_config = {
        "arbitrary_types_allowed": True
    }
    session: ClientSession | None = None
    llm_tools: list[dict[str, Any]] = Field(default_factory=list)


class McpServers(BaseModel):
    mcpServers: dict[str, McpServerConfig]


def load_config(file: Path) -> McpServers:
    if not file.exists():
        raise FileNotFoundError(f"Config file '{file}' not found.")
    config_text = ""
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip().rsplit("//", maxsplit=1)[0]  ## remove comments
        config_text += line + "\n"
    config = McpServers(**json.loads(config_text))
    for server in config.mcpServers.values():
        if not server.env:
            server.env = get_default_environment()
        for k, v in server.env.items():
            if v == "GET_ENV":
                server.env[k] = os.environ[k]  # adding environment variables from GET_ENV
        if "PATH" not in server.env:
            server.env["PATH"] = os.environ["PATH"]  # adding PATH helps MCP spawned process find things your path

    return config


mcp_servers: McpServers = load_config(config_file)
# console.print(server_servers)
# exit(0)

# Define MCP Server Parameters
# servers: list[StdioServerParameters] = [
#     # StdioServerParameters(
#     # ),
#     # StdioServerParameters(
#     #     command="uvx",
#     #     args=["mcp-server-fetch", "--ignore-robots-txt", "--user-agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"],
#     #     env={
#     #         "PATH": os.environ.get("PATH"),  # adding PATH helps MCP spawned process find things your path
#     #     },
#     # ),
#     # StdioServerParameters(
#     #     command="npx",
#     #     args=["-y", "@modelcontextprotocol/server-memory@latest"],
#     #     env={
#     #         "PATH": os.environ.get("PATH"),  # adding PATH helps MCP spawned process find things your path
#     #     },
#     # ),
# ]


# Works with any tool capable LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", timeout=10.0) # type:ignore
# llm = ChatOpenAI(model="gpt-4o")
# llm = ChatOllama(model="llama3.2:latest")


async def amain():
    """Async main function to connect to MCP."""
    llm_tools: list[dict] = []
    tool_nodes: list[McpToolNode] = []

    async with AsyncExitStack() as stack:
        for server_name, server_params in mcp_servers.mcpServers.items():
            try:
                console.print(
                    f"Connecting to MCP server {server_name} {server_params.command + ' ' + (' '.join(server_params.args))}..."
                )
                client = await stack.enter_async_context(stdio_client(server_params))
                # console.print(f"Starting session on MCP server {server_params.command}...")
                session = await stack.enter_async_context(ClientSession(*client))
                await session.initialize()
                server_params.session = session
                # console.print("Getting list of tools...")
                server_tools = await mcp_tool_list(session)
                tool_node = mcp_tool_node(session, server_tools)
                # tool_node = McpToolNode(session, handle_tool_errors=False)
                server_params.llm_tools = server_tools
                llm_tools.extend(server_tools)
                # console.print(f"MCP Tools from {server_name}:", server_tools)
                console.print(f"MCP Tools from {server_name}:", [tool["name"] for tool in server_tools])

                # await tool_node.init_funcs()
                tool_nodes.append(tool_node)
            except Exception as e:
                console.print(
                    f"Failed to connect to MCP server {server_name} {server_params.command + ' '  + (' '.join(server_params.args))}:",
                    e,
                )
                exit(1)
                # continue

        console.print("Connected to MCP servers.")

        if not llm_tools:
            console.print("No MCP tools available from any server.")
            return

        llm_with_tools = llm.bind_tools(llm_tools)
        sys_msg = SystemMessage(content="You are a helpful assistant. Use available tools to assist the user.")

        # Graph Node
        def assistant(state: MessagesState):
            return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

        # Build graph
        builder = StateGraph(MessagesState)
        # Define nodes: these do the work
        builder.add_node("assistant", assistant)

        # Combine all tool nodes into a single node

        builder.add_node("tools", combined_tool_node(tool_nodes, llm_tools))

        # Define edges: these determine how the control flow moves
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        react_graph = builder.compile()

        messages = [
            HumanMessage(
                # content="Search for Paul Robello the Principal Solution Architect and give me the current time"
                # content = "Search for Paul Robello the Principal Solution Architect"
                content="summarize the contents of url https://par-com.net/"
            )
        ]
        # Invoke the graph with initial messages
        messages = await react_graph.ainvoke({"messages": messages})
        for m in messages["messages"]:
            m.pretty_print()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
