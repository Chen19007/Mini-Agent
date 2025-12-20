"""ACP (Agent Client Protocol) bridge for Mini-Agent."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    AgentSideConnection,
    CancelNotification,
    InitializeRequest,
    InitializeResponse,
    LoadSessionRequest,
    LoadSessionResponse,
    NewSessionRequest,
    NewSessionResponse,
    PromptRequest,
    PromptResponse,
    session_notification,
    start_tool_call,
    stdio_streams,
    text_block,
    tool_content,
    update_agent_message,
    update_agent_thought,
    update_tool_call,
)
from pydantic import field_validator
from acp.schema import (
    AgentCapabilities,
    CurrentModeUpdate,
    Implementation,
    ListSessionsRequest,
    ListSessionsResponse,
    McpCapabilities,
    PermissionOption,
    PermissionOptionKind,
    SelectedPermissionOutcome,
    SessionCapabilities,
    SessionInfo,
    SessionListCapabilities,
    SessionMode,
    SessionModeState,
    SetSessionModeRequest,
    SetSessionModeResponse,
    ToolCallUpdate,
    ToolCallStatus,
    ToolKind,
)

from mini_agent.agent import Agent
from mini_agent.cli import add_workspace_tools, initialize_base_tools
from mini_agent.config import Config
from mini_agent.llm import LLMClient
from mini_agent.retry import RetryConfig as RetryConfigBase
from mini_agent.schema import Message
from mini_agent.tools.mcp_loader import get_mcp_tools_metadata_prompt, MCPTool
from mini_agent.acp.thread_storage import ThreadStorage
from mini_agent.acp.allow_list import AllowListStorage

logger = logging.getLogger(__name__)

# Mode constants
MODE_ASK = "ask"
MODE_AGENT = "agent"


try:
    class InitializeRequestPatch(InitializeRequest):
        @field_validator("protocolVersion", mode="before")
        @classmethod
        def normalize_protocol_version(cls, value: Any) -> int:
            if isinstance(value, str):
                try:
                    return int(value.split(".")[0])
                except Exception:
                    return 1
            if isinstance(value, (int, float)):
                return int(value)
            return 1

    InitializeRequest = InitializeRequestPatch
    InitializeRequest.model_rebuild(force=True)
except Exception:  # pragma: no cover - defensive
    logger.debug("ACP schema patch skipped")


@dataclass
class SessionState:
    agent: Agent
    cancelled: bool = False
    current_mode: str = MODE_AGENT  # Current session mode, default to agent
    thread_storage: ThreadStorage | None = None  # Thread storage for saving/loading session history
    allow_list_storage: AllowListStorage | None = None  # Allow list storage for MCP tool permissions


class MiniMaxACPAgent:
    """Minimal ACP adapter wrapping the existing Agent runtime."""

    def __init__(
        self,
        conn: AgentSideConnection,
        config: Config,
        llm: LLMClient,
        base_tools: list,
        system_prompt: str,
    ):
        self._conn = conn
        self._config = config
        self._llm = llm
        self._base_tools = base_tools
        self._system_prompt = system_prompt
        self._sessions: dict[str, SessionState] = {}

    def _get_ask_mode_prompt(self, base_prompt: str) -> str:
        """Generate system prompt for ask mode with clear mode restrictions.
        
        Args:
            base_prompt: Base system prompt (contains full capabilities)
            
        Returns:
            System prompt for ask mode with mode restrictions clearly stated
        """
        mode_header = """## Session Mode: ASK (Read-Only)

You are operating in **ASK mode**, which has the following restrictions:

### Tool Restrictions
- ✅ **Allowed**: Read-only tools
  - `read_file` - Read file contents
  - `search`, `parallel_search` - Web search (from MCP tools)
  - `browse` - Web browsing (from MCP tools)
  - Other read-only MCP tools
- ❌ **Restricted**: File modification tools
  - `write_file`, `edit_file`, `delete_file` - Not available in ASK mode
- ❌ **Restricted**: Command execution
  - `bash` - Not available in ASK mode

### Purpose
Answer questions and provide information without modifying files or executing commands.

When you need to read files or search the web to answer questions, you may use the appropriate read-only tools listed above.

---

"""
        # Combine mode restrictions with base capabilities
        # The base_prompt contains full capability descriptions, which helps LLM
        # understand what tools exist, but the mode restrictions above take precedence
        return mode_header + base_prompt

    def _get_agent_mode_prompt(self, base_prompt: str) -> str:
        """Generate system prompt for agent mode (full capabilities).
        
        Args:
            base_prompt: Base system prompt
            
        Returns:
            System prompt for agent mode (same as base, but with mode header for clarity)
        """
        mode_header = """## Session Mode: AGENT (Full Capabilities)

You are operating in **AGENT mode** with full access to all available tools:
- ✅ File operations (read, write, edit, delete)
- ✅ Command execution (bash)
- ✅ MCP tools (web search, browse, memory, etc.)

You can perform any operations needed to complete tasks.

---

"""
        return mode_header + base_prompt

    def _get_available_modes(self) -> list[SessionMode]:
        """Get list of available session modes.
        
        Returns:
            List of SessionMode objects
        """
        return [
            SessionMode(
                id=MODE_AGENT,
                name="Agent",
                description="Full agent mode with all tools enabled",
            ),
            SessionMode(
                id=MODE_ASK,
                name="Ask",
                description="Answer questions only, read-only tools allowed",
            ),
        ]

    def _get_thread_storage_dir(self) -> Path:
        """Get thread storage directory based on configuration.
        
        Returns:
            Path to thread storage directory
        """
        if self._config.agent.thread_storage_dir:
            return Path(self._config.agent.thread_storage_dir).expanduser().resolve()
        else:
            # Use config_dir / "threads" as default
            config_dir = getattr(self._config, "_config_dir", None)
            if config_dir:
                return Path(config_dir) / "threads"
            else:
                # Fallback to ~/.mini-agent/threads
                return Path.home() / ".mini-agent" / "threads"

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:  # noqa: ARG002
        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentCapabilities=AgentCapabilities(
                loadSession=True,
                sessionCapabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                ),
            ),
            agentInfo=Implementation(name="mini-agent", title="Mini-Agent", version="0.1.0"),
        )

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        session_id = f"sess-{len(self._sessions)}-{uuid4().hex[:8]}"
        workspace = Path(params.cwd or self._config.agent.workspace_dir).expanduser()
        if not workspace.is_absolute():
            workspace = workspace.resolve()
        tools = list(self._base_tools)
        add_workspace_tools(tools, self._config, workspace)
        agent = Agent(
            llm_client=self._llm,
            system_prompt=self._system_prompt,
            tools=tools,
            max_steps=self._config.agent.max_steps,
            workspace_dir=str(workspace),
            log_dir=self._config.agent.log_dir,
        )
        
        # Initialize thread storage
        thread_storage_dir = self._get_thread_storage_dir()
        thread_storage = ThreadStorage(thread_storage_dir)
        
        # Initialize allow list storage
        allow_list_storage = AllowListStorage(thread_storage_dir)
        
        self._sessions[session_id] = SessionState(
            agent=agent, 
            thread_storage=thread_storage,
            allow_list_storage=allow_list_storage
        )
        
        # Return available modes in the response
        available_modes = self._get_available_modes()
        return NewSessionResponse(
            sessionId=session_id,
            modes=SessionModeState(
                available_modes=available_modes,
                current_mode_id=MODE_AGENT,  # Default to agent mode
            ),
        )

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        state = self._sessions.get(params.sessionId)
        if not state:
            return PromptResponse(stopReason="refusal")
        state.cancelled = False
        
        # Adjust system prompt based on current mode
        # Both modes get a mode-specific header + base prompt for clarity
        if state.agent.messages and state.agent.messages[0].role == "system":
            if state.current_mode == MODE_ASK:
                # For ask mode, use ask mode prompt with restrictions
                ask_prompt = self._get_ask_mode_prompt(self._system_prompt)
                state.agent.messages[0].content = ask_prompt
            else:
                # For agent mode, use agent mode prompt with full capabilities
                agent_prompt = self._get_agent_mode_prompt(self._system_prompt)
                state.agent.messages[0].content = agent_prompt
        else:
            # If no system message exists, add one based on current mode
            if state.current_mode == MODE_ASK:
                ask_prompt = self._get_ask_mode_prompt(self._system_prompt)
                state.agent.messages.insert(0, Message(role="system", content=ask_prompt))
            else:
                agent_prompt = self._get_agent_mode_prompt(self._system_prompt)
                state.agent.messages.insert(0, Message(role="system", content=agent_prompt))
        
        user_text = "\n".join(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "") for block in params.prompt)
        state.agent.messages.append(Message(role="user", content=user_text))
        # Start logging for this ACP session turn
        state.agent.logger.start_new_run()
        stop_reason = await self._run_turn(state, params.sessionId)
        
        # Auto-save thread after each turn
        if state.thread_storage:
            try:
                workspace_dir = str(state.agent.workspace_dir)
                state.thread_storage.save_thread(
                    thread_id=params.sessionId,
                    messages=state.agent.messages,
                    workspace_dir=workspace_dir,
                )
            except Exception as exc:
                logger.warning(f"Failed to save thread {params.sessionId}: {exc}")
        
        return PromptResponse(stopReason=stop_reason)

    async def listSessions(self, params: ListSessionsRequest) -> ListSessionsResponse:
        """List all saved sessions/threads.
        
        Args:
            params: ListSessionsRequest (may contain workspace_dir filter)
            
        Returns:
            ListSessionsResponse with list of SessionInfo
        """
        from datetime import datetime
        
        logger.info(f"listSessions called with params: cwd={getattr(params, 'cwd', None)}, cursor={getattr(params, 'cursor', None)}")
        
        # Get thread storage directory
        thread_storage_dir = self._get_thread_storage_dir()
        thread_storage = ThreadStorage(thread_storage_dir)
        
        # Get workspace_dir from request if provided
        # Try multiple possible field names
        workspace_dir = None
        if hasattr(params, "workspace_dir") and params.workspace_dir:
            workspace_dir = str(params.workspace_dir)
        elif hasattr(params, "cwd") and params.cwd:
            workspace_dir = str(params.cwd)
        
        # Normalize workspace_dir path for comparison
        if workspace_dir:
            workspace_dir = str(Path(workspace_dir).resolve())
            logger.info(f"listSessions: filtering by workspace_dir={workspace_dir}")
        else:
            logger.info("listSessions: no workspace_dir filter, returning all threads")
        
        # List threads from storage (if workspace_dir is None, returns all threads)
        threads = thread_storage.list_threads(workspace_dir=workspace_dir)
        logger.info(f"listSessions: found {len(threads)} threads from storage")
        
        # Convert to SessionInfo objects
        session_infos = []
        for thread in threads:
            # Convert timestamp to ISO 8601 format
            updated_at = None
            if thread.get("updated_at"):
                try:
                    dt = datetime.fromtimestamp(thread["updated_at"])
                    updated_at = dt.isoformat()
                except (ValueError, TypeError, OSError):
                    updated_at = None
            
            # Normalize workspace_dir path for output
            thread_workspace = str(Path(thread["workspace_dir"]).resolve())
            
            session_infos.append(SessionInfo(
                sessionId=thread["id"],
                cwd=thread_workspace,
                title=thread.get("title") or "Untitled Thread",
                updatedAt=updated_at,
            ))
        
        logger.info(f"listSessions: returning {len(session_infos)} sessions")
        return ListSessionsResponse(sessions=session_infos)

    async def loadSession(self, params: LoadSessionRequest) -> LoadSessionResponse:
        """Load a saved session from thread storage.
        
        Args:
            params: LoadSessionRequest with sessionId
            
        Returns:
            LoadSessionResponse indicating success or failure
        """
        session_id = params.sessionId
        
        # Get thread storage directory
        thread_storage_dir = self._get_thread_storage_dir()
        thread_storage = ThreadStorage(thread_storage_dir)
        
        # Load thread from storage
        result = thread_storage.load_thread(session_id)
        if result is None:
            return LoadSessionResponse(success=False, error=f"Thread {session_id} not found")
        
        messages, workspace_dir = result
        
        if not messages:
            return LoadSessionResponse(success=False, error=f"Thread {session_id} has no messages")
        
        # Create workspace path
        workspace = Path(workspace_dir).expanduser()
        if not workspace.is_absolute():
            workspace = workspace.resolve()
        
        # Initialize tools
        tools = list(self._base_tools)
        add_workspace_tools(tools, self._config, workspace)
        
        # Create agent with loaded messages
        agent = Agent(
            llm_client=self._llm,
            system_prompt=self._system_prompt,
            tools=tools,
            max_steps=self._config.agent.max_steps,
            workspace_dir=str(workspace),
            log_dir=self._config.agent.log_dir,
        )
        
        # Restore messages (replace the default system message)
        agent.messages = messages
        
        # Initialize allow list storage
        allow_list_storage = AllowListStorage(thread_storage_dir)
        
        # Create session state
        state = SessionState(
            agent=agent, 
            thread_storage=thread_storage,
            allow_list_storage=allow_list_storage
        )
        self._sessions[session_id] = state
        
        # Stream historical messages back to client
        try:
            for msg in messages:
                if msg.role == "assistant":
                    if msg.thinking:
                        await self._send(session_id, update_agent_thought(text_block(msg.thinking)))
                    if msg.content:
                        await self._send(session_id, update_agent_message(text_block(msg.content)))
                elif msg.role == "user":
                    # User messages are typically not streamed back, but we can send them if needed
                    pass
        except Exception as exc:
            logger.warning(f"Failed to stream historical messages: {exc}")
        
        return LoadSessionResponse(success=True)

    async def cancel(self, params: CancelNotification) -> None:
        state = self._sessions.get(params.sessionId)
        if state:
            state.cancelled = True

    async def setSessionMode(self, params: SetSessionModeRequest) -> SetSessionModeResponse:
        """Handle session mode switching request.
        
        Args:
            params: SetSessionModeRequest with session_id and mode_id
            
        Returns:
            SetSessionModeResponse indicating success or failure
        """
        state = self._sessions.get(params.session_id)
        if not state:
            return SetSessionModeResponse(success=False, error="Session not found")
        
        # Validate mode ID
        valid_modes = [MODE_ASK, MODE_AGENT]
        if params.mode_id not in valid_modes:
            return SetSessionModeResponse(
                success=False,
                error=f"Invalid mode: {params.mode_id}. Valid modes are: {', '.join(valid_modes)}",
            )
        
        # Update mode (will take effect on next prompt call)
        state.current_mode = params.mode_id
        
        # Send mode update notification
        try:
            await self._send(params.session_id, CurrentModeUpdate(currentModeId=params.mode_id))
        except Exception as exc:
            logger.warning(f"Failed to send mode update notification: {exc}")
        
        return SetSessionModeResponse(success=True)

    def _requires_confirmation(self, tool_name: str) -> bool:
        """Check if a tool requires user confirmation before execution.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool requires confirmation, False otherwise
        """
        # Tools that require confirmation due to potential danger
        dangerous_tools = {
            "bash",  # Execute arbitrary system commands
            "write_file",  # May overwrite existing files
            "edit_file",  # Modify file content
            "delete_file",  # Delete files
        }
        return tool_name in dangerous_tools

    def _is_delete_operation(self, tool_name: str) -> bool:
        """Check if a tool is a delete operation.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if tool is a delete operation, False otherwise
        """
        return "delete" in tool_name.lower()

    async def _request_tool_permission(
        self,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_label: str,
        tool: Any = None,
    ) -> tuple[bool, str]:
        """Request permission from the client to execute a tool.
        
        Args:
            session_id: Current session ID
            tool_call_id: ID of the tool call
            tool_name: Name of the tool
            tool_args: Arguments for the tool
            tool_label: Display label for the tool call
            tool: Tool object (optional, used to determine if it's an MCP tool)
            
        Returns:
            Tuple of (permission_granted: bool, option_id: str)
            - permission_granted: True if permission is granted, False otherwise
            - option_id: The selected option ID ("allow_once", "add_to_allow_list", "reject_once")
        """
        try:
            # Determine tool kind based on tool name
            # ToolKind is a Literal type, so we use string values directly
            tool_kind = "execute"  # Default for bash and other execution tools
            if tool_name in ("read_file",):
                tool_kind = "read"
            elif tool_name in ("write_file", "edit_file", "delete_file"):
                tool_kind = "edit"  # Use "edit" for write operations
            
            # Build tool call update
            # ToolCallStatus is also a Literal type, so we use string values directly
            tool_call_update = ToolCallUpdate(
                tool_call_id=tool_call_id,
                title=tool_label,
                kind=tool_kind,
                status="pending",
                raw_input=tool_args,
            )
            
            # Build permission options
            # PermissionOptionKind is also a Literal type, so we use string values directly
            options = [
                PermissionOption(
                    option_id="allow_once",
                    name="Allow once",
                    kind="allow_once",
                ),
                PermissionOption(
                    option_id="reject_once",
                    name="Reject",
                    kind="reject_once",
                ),
            ]
            
            # Add "Add to allow list" option only for MCP tools
            if tool is not None and isinstance(tool, MCPTool):
                options.insert(1, PermissionOption(
                    option_id="add_to_allow_list",
                    name="Add to allow list",
                    kind="allow_once",  # Use allow_once kind for compatibility
                ))
            
            # Request permission - pass parameters directly, not as a RequestPermissionRequest object
            response = await self._conn.request_permission(
                options=options,
                session_id=session_id,
                tool_call=tool_call_update,
            )
            
            # Check if permission was granted and get the selected option
            # The response has an 'outcome' field which is a SelectedPermissionOutcome
            # with an 'option_id' field indicating which option was chosen
            option_id = "reject_once"  # Default to reject
            if hasattr(response, "outcome") and response.outcome is not None:
                if isinstance(response.outcome, SelectedPermissionOutcome):
                    option_id = response.outcome.option_id
                # Handle case where outcome might be a dict
                elif isinstance(response.outcome, dict):
                    option_id = response.outcome.get("option_id", "reject_once")
            
            # Determine if permission was granted
            permission_granted = option_id in ("allow_once", "add_to_allow_list")
            
            return permission_granted, option_id
        except Exception as exc:
            logger.exception(f"Error requesting permission for tool {tool_name}: {exc}")
            # On error, default to denying permission for safety
            return False, "reject_once"

    async def _run_turn(self, state: SessionState, session_id: str) -> str:
        agent = state.agent
        
        # Define read-only tools allowed in ask mode
        read_only_tools = {"read_file"}  # Built-in read-only tools
        
        # Determine available tools based on mode
        if state.current_mode == MODE_ASK:
            # Ask mode: allow read-only tools and all MCP tools (search, browse, etc.)
            available_tools = {
                name: tool 
                for name, tool in agent.tools.items() 
                if name in read_only_tools or isinstance(tool, MCPTool)
            }
            tool_schemas = [tool.to_schema() for tool in available_tools.values()]
            tool_list = list(available_tools.values())
        else:
            # Agent mode: all tools
            tool_schemas = [tool.to_schema() for tool in agent.tools.values()]
            tool_list = list(agent.tools.values())
            available_tools = agent.tools
        
        for _ in range(agent.max_steps):
            if state.cancelled:
                return "cancelled"
            # Log LLM request
            agent.logger.log_request(messages=agent.messages, tools=tool_list)
            try:
                response = await agent.llm.generate(messages=agent.messages, tools=tool_schemas)
            except Exception as exc:
                logger.exception("LLM error")
                await self._send(session_id, update_agent_message(text_block(f"Error: {exc}")))
                return "refusal"
            # Log LLM response
            agent.logger.log_response(
                content=response.content,
                thinking=response.thinking,
                tool_calls=response.tool_calls,
                finish_reason=response.finish_reason,
            )
            if response.thinking:
                await self._send(session_id, update_agent_thought(text_block(response.thinking)))
            if response.content:
                await self._send(session_id, update_agent_message(text_block(response.content)))
            agent.messages.append(Message(role="assistant", content=response.content, thinking=response.thinking, tool_calls=response.tool_calls))
            if not response.tool_calls:
                return "end_turn"
            for call in response.tool_calls:
                name, args = call.function.name, call.function.arguments
                
                # In ask mode, reject tools that are not in the allowed list
                if state.current_mode == MODE_ASK and name not in available_tools:
                    text = f"❌ Tool '{name}' is not available in ASK mode. Only read-only tools (like read_file) and MCP tools (like search, browse) are allowed."
                    status = "failed"
                    # Show tool call attempt
                    args_preview = ", ".join(f"{k}={repr(v)[:50]}" for k, v in list(args.items())[:2]) if isinstance(args, dict) else ""
                    label = f"🔧 {name}({args_preview})" if args_preview else f"🔧 {name}()"
                    await self._send(session_id, start_tool_call(call.id, label, kind="execute", raw_input=args))
                    # Log tool execution result
                    agent.logger.log_tool_result(
                        tool_name=name,
                        arguments=args,
                        result_success=False,
                        result_error=text,
                    )
                    await self._send(session_id, update_tool_call(call.id, status=status, content=[tool_content(text_block(text))], raw_output=text))
                    agent.messages.append(Message(role="tool", content=text, tool_call_id=call.id, name=name))
                    continue
                
                # Show tool name with key arguments for better visibility
                args_preview = ", ".join(f"{k}={repr(v)[:50]}" for k, v in list(args.items())[:2]) if isinstance(args, dict) else ""
                label = f"🔧 {name}({args_preview})" if args_preview else f"🔧 {name}()"
                await self._send(session_id, start_tool_call(call.id, label, kind="execute", raw_input=args))
                tool = available_tools.get(name)
                if not tool:
                    text, status = f"❌ Unknown tool: {name}", "failed"
                    # Log tool execution result
                    agent.logger.log_tool_result(
                        tool_name=name,
                        arguments=args,
                        result_success=False,
                        result_error=text,
                    )
                else:
                    # Check allow list for MCP tools (skip confirmation if allowed and not a delete operation)
                    permission_granted = True
                    option_id = "allow_once"
                    
                    # Check if tool is in allow list and not a delete operation
                    if isinstance(tool, MCPTool) and state.allow_list_storage:
                        if state.allow_list_storage.is_allowed(name):
                            # Tool is in allow list, but still require confirmation for delete operations
                            if not self._is_delete_operation(name):
                                # Auto-allow non-delete operations from allow list
                                permission_granted = True
                            else:
                                # Delete operations still need confirmation even if in allow list
                                permission_granted, option_id = await self._request_tool_permission(
                                    session_id=session_id,
                                    tool_call_id=call.id,
                                    tool_name=name,
                                    tool_args=args,
                                    tool_label=label,
                                    tool=tool,
                                )
                        else:
                            # MCP tool not in allow list - always request permission (to show add_to_allow_list option)
                            permission_granted, option_id = await self._request_tool_permission(
                                session_id=session_id,
                                tool_call_id=call.id,
                                tool_name=name,
                                tool_args=args,
                                tool_label=label,
                                tool=tool,
                            )
                    elif self._requires_confirmation(name):
                        # Non-MCP tool that requires confirmation (e.g., bash)
                        permission_granted, option_id = await self._request_tool_permission(
                            session_id=session_id,
                            tool_call_id=call.id,
                            tool_name=name,
                            tool_args=args,
                            tool_label=label,
                            tool=tool,
                        )
                    
                    # Handle add_to_allow_list option
                    if permission_granted and option_id == "add_to_allow_list" and state.allow_list_storage:
                        state.allow_list_storage.add_tool(name)
                        logger.info(f"Added tool '{name}' to allow list")
                    
                    if not permission_granted:
                        text, status = "❌ Permission denied by user", "failed"
                        # Log tool execution result
                        agent.logger.log_tool_result(
                            tool_name=name,
                            arguments=args,
                            result_success=False,
                            result_error=text,
                        )
                    else:
                        # Permission granted, proceed with execution
                        try:
                            result = await tool.execute(**args)
                            status = "completed" if result.success else "failed"
                            prefix = "✅" if result.success else "❌"
                            text = f"{prefix} {result.content if result.success else result.error or 'Tool execution failed'}"
                            # Log tool execution result
                            agent.logger.log_tool_result(
                                tool_name=name,
                                arguments=args,
                                result_success=result.success,
                                result_content=result.content if result.success else None,
                                result_error=result.error if not result.success else None,
                            )
                        except Exception as exc:
                            status, text = "failed", f"❌ Tool error: {exc}"
                            # Log tool execution result
                            agent.logger.log_tool_result(
                                tool_name=name,
                                arguments=args,
                                result_success=False,
                                result_error=str(exc),
                            )
                await self._send(session_id, update_tool_call(call.id, status=status, content=[tool_content(text_block(text))], raw_output=text))
                agent.messages.append(Message(role="tool", content=text, tool_call_id=call.id, name=name))
        return "max_turn_requests"

    async def _send(self, session_id: str, update: Any) -> None:
        await self._conn.sessionUpdate(session_notification(session_id, update))


async def run_acp_server(config: Config | None = None, config_path: Path | str | None = None) -> None:
    """Run Mini-Agent as an ACP-compatible stdio server.
    
    Args:
        config: Optional Config instance. If provided, config_path is ignored.
        config_path: Optional configuration file path. If None and config is None, uses default search path.
    """
    config_source = "provided Config instance"
    actual_config_path = None
    
    if config is None:
        if config_path is None:
            # Check environment variable
            import os
            env_config_path = os.environ.get("MINI_AGENT_CONFIG_PATH")
            if env_config_path:
                config_path = env_config_path
                config_source = f"environment variable MINI_AGENT_CONFIG_PATH: {config_path}"
            else:
                config_path = Config.get_default_config_path()
                config_source = f"default search path: {config_path}"
            actual_config_path = config_path
            config = Config.from_yaml(config_path)
        else:
            config_path = Path(config_path).expanduser().resolve()
            actual_config_path = config_path
            config_source = f"provided config_path: {config_path}"
            config = Config.from_yaml(config_path)
    else:
        # Config instance provided, try to determine source
        config_dir = getattr(config, "_config_dir", None)
        if config_dir:
            actual_config_path = Path(config_dir) / "config.yaml"
            if actual_config_path.exists():
                config_source = f"provided Config instance (from: {actual_config_path})"
            else:
                config_source = "provided Config instance (path unknown)"
        else:
            config_source = "provided Config instance (path unknown)"
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    # Log configuration paths
    logger.info("=" * 60)
    logger.info("Mini-Agent ACP Server - Configuration Paths")
    logger.info("=" * 60)
    logger.info(f"Config file: {actual_config_path} ({config_source})")
    
    config_dir = getattr(config, "_config_dir", None)
    if config_dir:
        logger.info(f"Config directory: {config_dir}")
    
    # Workspace directory (from config or default)
    workspace_dir = Path(config.agent.workspace_dir).expanduser().resolve()
    logger.info(f"Workspace directory: {workspace_dir}")
    
    # Log directory
    if config.agent.log_dir:
        log_dir_resolved = Path(config.agent.log_dir).expanduser().resolve()
        logger.info(f"Log directory: {log_dir_resolved} (from config)")
    else:
        default_log_dir = Path.home() / ".mini-agent" / "log"
        logger.info(f"Log directory: {default_log_dir} (default)")
    
    # Thread storage directory
    if config.agent.thread_storage_dir:
        thread_storage_dir_resolved = Path(config.agent.thread_storage_dir).expanduser().resolve()
        logger.info(f"Thread storage directory: {thread_storage_dir_resolved} (from config)")
    else:
        config_dir = getattr(config, "_config_dir", None)
        if config_dir:
            default_thread_storage_dir = Path(config_dir) / "threads"
        else:
            default_thread_storage_dir = Path.home() / ".mini-agent" / "threads"
        logger.info(f"Thread storage directory: {default_thread_storage_dir} (default)")
    
    # System prompt path
    prompt_path = config.find_config_file(config.agent.system_prompt_path)
    if prompt_path and prompt_path.exists():
        logger.info(f"System prompt: {prompt_path}")
    else:
        logger.info(f"System prompt: {config.agent.system_prompt_path} (not found, using default)")
    
    # MCP config path
    if config.tools.enable_mcp:
        mcp_config_path = config.find_config_file(config.tools.mcp_config_path)
        if mcp_config_path and mcp_config_path.exists():
            logger.info(f"MCP config: {mcp_config_path}")
        else:
            logger.info(f"MCP config: {config.tools.mcp_config_path} (not found)")
    
    logger.info("=" * 60)
    
    base_tools = await initialize_base_tools(config)
    
    # Log enabled MCP servers and tools
    if config.tools.enable_mcp:
        mcp_config_path = config.find_config_file(config.tools.mcp_config_path)
        if mcp_config_path and mcp_config_path.exists():
            try:
                with open(mcp_config_path, encoding="utf-8") as f:
                    mcp_config = json.load(f)
                mcp_servers = mcp_config.get("mcpServers", {})
                enabled_servers = []
                disabled_servers = []
                for server_name, server_config in mcp_servers.items():
                    if server_config.get("disabled", False):
                        disabled_servers.append(server_name)
                    else:
                        enabled_servers.append(server_name)
                
                if enabled_servers:
                    logger.info(f"Enabled MCP servers ({len(enabled_servers)}): {', '.join(enabled_servers)}")
                if disabled_servers:
                    logger.info(f"Disabled MCP servers ({len(disabled_servers)}): {', '.join(disabled_servers)}")
            except Exception as e:
                logger.warning(f"Failed to parse MCP config for logging: {e}")
    
    # Extract MCP tools and log them
    mcp_tools = [tool for tool in base_tools if isinstance(tool, MCPTool)]
    if mcp_tools:
        # Group tools by server (if we can determine which server they came from)
        logger.info(f"Loaded {len(mcp_tools)} MCP tools:")
        for tool in mcp_tools:
            logger.info(f"  - {tool.name}: {tool.description[:80] if len(tool.description) > 80 else tool.description}")
    else:
        if config.tools.enable_mcp:
            logger.info("No MCP tools loaded (check MCP server connections)")
    
    if prompt_path and prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = "You are a helpful AI assistant."
    
    # Remove Skills Metadata placeholder (skills feature removed)
    system_prompt = system_prompt.replace("{SKILLS_METADATA}", "")
    
    # Inject MCP Tools Metadata into System Prompt
    if mcp_tools:
        mcp_metadata = get_mcp_tools_metadata_prompt(mcp_tools)
        if mcp_metadata:
            # Replace placeholder with actual metadata
            system_prompt = system_prompt.replace("{MCP_TOOLS_METADATA}", mcp_metadata)
            logger.info(f"Injected {len(mcp_tools)} MCP tools metadata into system prompt")
        else:
            # Remove placeholder if no metadata generated
            system_prompt = system_prompt.replace("{MCP_TOOLS_METADATA}", "")
    else:
        # Generate metadata even if no tools (will show a helpful message)
        mcp_metadata = get_mcp_tools_metadata_prompt([])
        system_prompt = system_prompt.replace("{MCP_TOOLS_METADATA}", mcp_metadata)
    
    # Check if file-edit-mcp-server tools are available
    # If not, remove the "File Operations" section from system prompt
    file_edit_tool_names = {
        "file_read", "file_write", "edit_file_by_context", 
        "file_insert_at_head", "file_append_at_tail", 
        "file_find_position", "directory_create", "file_delete"
    }
    has_file_edit_tools = any(tool.name in file_edit_tool_names for tool in mcp_tools) if mcp_tools else False
    
    if not has_file_edit_tools:
        # Remove the entire "File Operations" section
        # Match from "### File Operations" to the next "###" section (including the blank line before it)
        # Pattern: "### File Operations" followed by content until the next "###" section
        file_ops_pattern = r"### File Operations.*?(?=\n### |\Z)"
        system_prompt = re.sub(file_ops_pattern, "", system_prompt, flags=re.DOTALL)
        # Clean up any double newlines that might result from removal
        system_prompt = re.sub(r"\n\n\n+", "\n\n", system_prompt)
        # Also remove the reference to file-edit-mcp-server in "Core Capabilities" section
        system_prompt = system_prompt.replace(
            "- **MCP Tools**: Access additional tools from configured MCP servers (including file-edit-mcp-server for file operations)",
            "- **MCP Tools**: Access additional tools from configured MCP servers"
        )
        logger.info("Removed File Operations section from system prompt (file-edit-mcp-server tools not available)")
    rcfg = config.llm.retry
    llm = LLMClient(api_key=config.llm.api_key, api_base=config.llm.api_base, model=config.llm.model, retry_config=RetryConfigBase(enabled=rcfg.enabled, max_retries=rcfg.max_retries, initial_delay=rcfg.initial_delay, max_delay=rcfg.max_delay, exponential_base=rcfg.exponential_base))
    reader, writer = await stdio_streams()
    AgentSideConnection(lambda conn: MiniMaxACPAgent(conn, config, llm, base_tools, system_prompt), writer, reader)
    logger.info("Mini-Agent ACP server running")
    await asyncio.Event().wait()


def main() -> None:
    """Main entry point for ACP server.
    
    Supports MINI_AGENT_CONFIG_PATH environment variable to specify config file path.
    """
    import os
    config_path = os.environ.get("MINI_AGENT_CONFIG_PATH")
    asyncio.run(run_acp_server(config_path=config_path))


__all__ = ["MiniMaxACPAgent", "run_acp_server", "main"]
