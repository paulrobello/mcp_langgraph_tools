from __future__ import annotations

from typing import (
    Literal,
)
from collections.abc import Callable, Coroutine
from rich.console import Console
from langgraph.prebuilt.tool_node import (
    INVALID_TOOL_NAME_ERROR_TEMPLATE,
    _handle_tool_error,
    _infer_handled_types,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolCall,
    ToolMessage,
    HumanMessage,
    ChatMessage,
    SystemMessage,
    FunctionMessage,
    AIMessageChunk,
    HumanMessageChunk,
    ChatMessageChunk,
    SystemMessageChunk,
    FunctionMessageChunk,
    ToolMessageChunk,
)
from langchain_core.runnables import RunnableConfig

from langgraph.errors import GraphInterrupt
from mcp import ClientSession

from pydantic import BaseModel
from typing import Any


console = Console()

McpToolNode = Callable[[ToolCall, RunnableConfig | None], Coroutine[Any, Any, ToolMessage | None]]


def mcp_tool_node(
    session: ClientSession,
    llm_tools: list[dict],
    *,
    handle_tool_errors: bool | str | Callable[..., str] | tuple[type[Exception], ...] = True,
    whitelisted_tools: list[str] | None = None,
    blacklisted_tools: list[str] | None = None,
) -> McpToolNode:
    """Basic tool node that makes calls to MCP tools."""
    _valid_tools = []
    for tool in llm_tools:
        if whitelisted_tools and tool["name"] not in whitelisted_tools:
            continue
        if blacklisted_tools and tool["name"] in blacklisted_tools:
            continue
        _valid_tools.append(tool["name"])

    async def my_tool_node(tool_call: ToolCall, config: RunnableConfig | None = None) -> ToolMessage | None:
        # console.print("+-+-+-+- Tool_call:", tool_call)
        # console.print("MyTools:", valid_tools)
        if tool_call["name"] not in _valid_tools:
            return None
        try:
            res = await session.call_tool(tool_call["name"], arguments=tool_call["args"])
            # console.print("+-+-+-+- Result:", res)
            return ToolMessage(
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                content=res.content,  # type:ignore
                status="error" if res.isError else "success",
            )
        except GraphInterrupt as e:
            raise e
        except Exception as e:
            if isinstance(handle_tool_errors, tuple):
                handled_types: tuple = handle_tool_errors
            elif callable(handle_tool_errors):
                handled_types = _infer_handled_types(handle_tool_errors)
            else:
                # default behavior is catching all exceptions
                handled_types = (Exception,)

            # Unhandled
            if not handle_tool_errors or not isinstance(e, handled_types):
                raise e
            # Handled
            else:
                content = _handle_tool_error(e, flag=handle_tool_errors)

        return ToolMessage(content=content, name=tool_call["name"], tool_call_id=tool_call["id"], status="error")

    return my_tool_node


async def mcp_tool_list(session: ClientSession) -> list[dict[str, Any]]:
    """Gets list of tools from MCP and converts to OpenAI standard schema."""
    try:
        res = await session.list_tools()
        mcp_tools = res.tools
    except Exception as _:
        mcp_tools = []
    console.print(mcp_tools)
    # map mcp tools to openai spec dict
    llm_tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }
        for tool in mcp_tools
        if isinstance(tool, BaseModel)
    ]
    return llm_tools


def parse_input(
    input: list[AnyMessage] | dict[str, Any] | BaseModel, messages_key: str = "messages"
) -> tuple[list[ToolCall], Literal["list", "dict"]]:
    if isinstance(input, list):
        output_type = "list"
        message: AnyMessage = input[-1]
    elif isinstance(input, dict) and (messages := input.get(messages_key, [])):
        output_type = "dict"
        message = messages[-1]
    elif messages := getattr(input, messages_key, None):
        # Assume dataclass-like state that can coerce from dict
        output_type = "dict"
        message = messages[-1]
    else:
        raise ValueError("No message found in input")

    if not isinstance(message, AIMessage):
        raise ValueError("Last message is not an AIMessage")
    return message.tool_calls, output_type


def combined_tool_node(tool_nodes: list[McpToolNode], llm_tools: list[dict]):
    async def tool_node(
        calls: list[
            AIMessage
            | HumanMessage
            | ChatMessage
            | SystemMessage
            | FunctionMessage
            | ToolMessage
            | AIMessageChunk
            | HumanMessageChunk
            | ChatMessageChunk
            | SystemMessageChunk
            | FunctionMessageChunk
            | ToolMessageChunk
        ]
        | dict[str, Any]
        | BaseModel,
        config: RunnableConfig | None = None,
    ) -> dict[str, list[ToolMessage]]:
        tool_calls, output_type = parse_input(calls)
        results = []
        for call in tool_calls:
            # console.print(call)
            for tool_node in tool_nodes:
                # if tool_node.has_tool(call):
                try:
                    result: ToolMessage | None = await tool_node(call, config)
                    if not result:
                        continue
                    results.append(result)
                    # console.print("TOOL RESULT START " * 5)
                    # console.print(result)
                    # console.print("TOOL RESULT END " * 5)
                except Exception as e:
                    console.print("TOOL EXCEPTION START " * 5)
                    console.print(e)
                    console.print("TOOL EXCEPTION END " * 5)
                    continue
            if not len(results) > 0:
                console.print("No tools found. " * 3)
                content = INVALID_TOOL_NAME_ERROR_TEMPLATE.format(
                    requested_tool=call["name"],
                    available_tools=", ".join([tool["name"] for tool in llm_tools]),
                )
                results.append(ToolMessage(content, name=call["name"], tool_call_id=call["id"], status="error"))
        # console.print("--------Results:", results)
        return {"messages": results}

    return tool_node
