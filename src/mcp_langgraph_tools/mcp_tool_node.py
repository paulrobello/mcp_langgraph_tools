from __future__ import annotations

import asyncio
from typing import (
    Literal,
    cast,
)
from collections.abc import Callable
from rich.console import Console
from langgraph.prebuilt.tool_node import (
    msg_content_output,
    INVALID_TOOL_NAME_ERROR_TEMPLATE,
    _handle_tool_error,
    _infer_handled_types,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input

from langgraph.errors import GraphInterrupt
from langgraph.store.base import BaseStore
from langgraph.utils.runnable import RunnableCallable
from mcp import ClientSession

from pydantic import BaseModel
from typing import Any

console = Console()


def mcp_tool_node_basic(session: ClientSession):
    """Basic tool node that makes calls to MCP tools."""

    async def my_tool_node(state: dict):
        result = []
        # console.print("state:", state)
        for tool_call in state["messages"][-1].tool_calls:
            # console.print("Tool calls:", tool_call)
            res = await session.call_tool(tool_call["name"], arguments=tool_call["args"])
            tool_message: ToolMessage = ToolMessage(
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                content=res.content,
                status="error" if res.isError else "success",
            )
            result.append(tool_message)
        return {"messages": result}

    return my_tool_node


async def mcp_tool_list(session: ClientSession) -> list[dict[str, Any]]:
    """Gets list of tools from MCP and converts to OpenAI standard schema."""
    try:
        mcp_tools = (await session.list_tools()).tools
    except Exception as _:
        mcp_tools = []
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


class McpToolNode(RunnableCallable):
    """A node that runs the tools called in the last AIMessage.

    It can be used either in StateGraph with a "messages" state key (or a custom key passed via ToolNode's 'messages_key').
    If multiple tool calls are requested, they will be run in parallel. The output will be
    a list of ToolMessages, one for each tool call.


    Args:
        mcp_session: An initialized MCP ClientSession.
        whitelisted_tools: A list of tool names that can be run. Defaults to None = Allow all
        blacklisted_tools: A list of tool names that should not be run. Defaults to None = Allow all
        name: The name of the ToolNode in the graph. Defaults to "tools".
        tags: Optional tags to associate with the node. Defaults to None.
        handle_tool_errors: How to handle tool errors raised by tools inside the node. Defaults to True.
            Must be one of the following:

            - True: all errors will be caught and
                a ToolMessage with a default error message (TOOL_CALL_ERROR_TEMPLATE) will be returned.
            - str: all errors will be caught and
                a ToolMessage with the string value of 'handle_tool_errors' will be returned.
            - tuple[type[Exception], ...]: exceptions in the tuple will be caught and
                a ToolMessage with a default error message (TOOL_CALL_ERROR_TEMPLATE) will be returned.
            - Callable[..., str]: exceptions from the signature of the callable will be caught and
                a ToolMessage with the string value of the result of the 'handle_tool_errors' callable will be returned.
            - False: none of the errors raised by the tools will be caught
        messages_key: The state key in the input that contains the list of messages.
            The same key will be used for the output from the ToolNode.
            Defaults to "messages".

    Important:
        - This node must me used in an async graph. graph.ainvoke()
        - Must be called before the first invocation to populate the tools_by_name dictionary.
        - The state MUST contain a list of messages.
        - The last message MUST be an `AIMessage`.
        - The `AIMessage` MUST have `tool_calls` populated.
    """

    name: str = "ToolNode"

    def __init__(
        self,
        mcp_session: ClientSession,
        *,
        whitelisted_tools: list[str] | None = None,
        blacklisted_tools: list[str] | None = None,
        name: str = "tools",
        tags: list[str] | None = None,
        handle_tool_errors: bool | str | Callable[..., str] | tuple[type[Exception], ...] = True,
        messages_key: str = "messages",
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.tools_by_name: dict[str, dict] = {}
        self.handle_tool_errors = handle_tool_errors
        self.messages_key = messages_key
        self.mcp_session = mcp_session
        self.whitelisted_tools = whitelisted_tools
        self.blacklisted_tools = blacklisted_tools

    async def init_funcs(self) -> McpToolNode:
        """Must be called before the first invocation to populate the tools_by_name dictionary."""
        llm_tools = await mcp_tool_list(self.mcp_session)

        for tool in llm_tools:
            if self.whitelisted_tools is not None and tool["name"] not in self.whitelisted_tools:
                continue
            if self.blacklisted_tools is not None and tool["name"] in self.blacklisted_tools:
                continue
            self.tools_by_name[tool["name"]] = tool
        return self

    def _func(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: RunnableConfig,
        *,
        store: BaseStore,
    ) -> Any:
        raise NotImplementedError("You must use _afunc")

    def invoke(self, input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        raise NotImplementedError("You must use ainvoke")

    async def ainvoke(self, input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        if "store" not in kwargs:
            kwargs["store"] = None
        return await super().ainvoke(input, config, **kwargs)

    async def _afunc(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: RunnableConfig,
        *,
        store: BaseStore,
    ) -> Any:
        tool_calls, output_type = self._parse_input(input, store)
        outputs = await asyncio.gather(*(self._arun_one(call, config) for call in tool_calls))
        # TypedDict, pydantic, dataclass, etc. should all be able to load from dict
        return outputs if output_type == "list" else {self.messages_key: outputs}

    def _run_one(self, call: ToolCall, config: RunnableConfig) -> ToolMessage:
        raise NotImplementedError("You must use _arun_one")

    async def _arun_one(self, call: ToolCall, config: RunnableConfig) -> ToolMessage:
        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message

        try:
            # console.print(call["args"])
            res = await self.mcp_session.call_tool(call["name"], arguments=call["args"])
            if res.isError:
                raise Exception(res.content)
            tool_message: ToolMessage = ToolMessage(name=call["name"], tool_call_id=call["id"], content=res.content)

            tool_message.content = cast(str | list, msg_content_output(tool_message.content))
            return tool_message
        # GraphInterrupt is a special exception that will always be raised.
        # It can be triggered in the following scenarios:
        # (1) a NodeInterrupt is raised inside a tool
        # (2) a NodeInterrupt is raised inside a graph node for a graph called as a tool
        # (3) a GraphInterrupt is raised when a subgraph is interrupted inside a graph called as a tool
        # (2 and 3 can happen in a "supervisor w/ tools" multi-agent architecture)
        except GraphInterrupt as e:
            raise e
        except Exception as e:
            if isinstance(self.handle_tool_errors, tuple):
                handled_types: tuple = self.handle_tool_errors
            elif callable(self.handle_tool_errors):
                handled_types = _infer_handled_types(self.handle_tool_errors)
            else:
                # default behavior is catching all exceptions
                handled_types = (Exception,)

            # Unhandled
            if not self.handle_tool_errors or not isinstance(e, handled_types):
                raise e
            # Handled
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_errors)

        return ToolMessage(content=content, name=call["name"], tool_call_id=call["id"], status="error")

    def _parse_input(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        store: BaseStore,
    ) -> tuple[list[ToolCall], Literal["list", "dict"]]:
        if isinstance(input, list):
            output_type = "list"
            message: AnyMessage = input[-1]
        elif isinstance(input, dict) and (messages := input.get(self.messages_key, [])):
            output_type = "dict"
            message = messages[-1]
        elif messages := getattr(input, self.messages_key, None):
            # Assume dataclass-like state that can coerce from dict
            output_type = "dict"
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")
        return message.tool_calls, output_type

    def _validate_tool_call(self, call: ToolCall) -> ToolMessage | None:
        if (requested_tool := call["name"]) not in self.tools_by_name:
            content = INVALID_TOOL_NAME_ERROR_TEMPLATE.format(
                requested_tool=requested_tool,
                available_tools=", ".join(self.tools_by_name.keys()),
            )
            return ToolMessage(content, name=requested_tool, tool_call_id=call["id"], status="error")
        else:
            return None
