"""ACP (Agent Client Protocol) bridge for Mini-Agent."""

from __future__ import annotations

import asyncio
import logging
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
    Implementation,
    McpCapabilities,
    PermissionOption,
    PermissionOptionKind,
    SelectedPermissionOutcome,
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

logger = logging.getLogger(__name__)


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

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:  # noqa: ARG002
        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentCapabilities=AgentCapabilities(loadSession=False),
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
        self._sessions[session_id] = SessionState(agent=agent)
        return NewSessionResponse(sessionId=session_id)

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        state = self._sessions.get(params.sessionId)
        if not state:
            return PromptResponse(stopReason="refusal")
        state.cancelled = False
        user_text = "\n".join(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "") for block in params.prompt)
        state.agent.messages.append(Message(role="user", content=user_text))
        # Start logging for this ACP session turn
        state.agent.logger.start_new_run()
        stop_reason = await self._run_turn(state, params.sessionId)
        return PromptResponse(stopReason=stop_reason)

    async def cancel(self, params: CancelNotification) -> None:
        state = self._sessions.get(params.sessionId)
        if state:
            state.cancelled = True

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

    async def _request_tool_permission(
        self,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_label: str,
    ) -> bool:
        """Request permission from the client to execute a tool.
        
        Args:
            session_id: Current session ID
            tool_call_id: ID of the tool call
            tool_name: Name of the tool
            tool_args: Arguments for the tool
            tool_label: Display label for the tool call
            
        Returns:
            True if permission is granted, False otherwise
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
            
            # Request permission - pass parameters directly, not as a RequestPermissionRequest object
            response = await self._conn.request_permission(
                options=options,
                session_id=session_id,
                tool_call=tool_call_update,
            )
            
            # Check if permission was granted
            # The response has an 'outcome' field which is a SelectedPermissionOutcome
            # with an 'option_id' field indicating which option was chosen
            if hasattr(response, "outcome") and response.outcome is not None:
                if isinstance(response.outcome, SelectedPermissionOutcome):
                    return response.outcome.option_id == "allow_once"
                # Handle case where outcome might be a dict
                if isinstance(response.outcome, dict):
                    return response.outcome.get("option_id") == "allow_once"
            # Default to False if we can't determine
            return False
        except Exception as exc:
            logger.exception(f"Error requesting permission for tool {tool_name}: {exc}")
            # On error, default to denying permission for safety
            return False

    async def _run_turn(self, state: SessionState, session_id: str) -> str:
        agent = state.agent
        for _ in range(agent.max_steps):
            if state.cancelled:
                return "cancelled"
            tool_schemas = [tool.to_schema() for tool in agent.tools.values()]
            # Log LLM request
            tool_list = list(agent.tools.values())
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
                # Show tool name with key arguments for better visibility
                args_preview = ", ".join(f"{k}={repr(v)[:50]}" for k, v in list(args.items())[:2]) if isinstance(args, dict) else ""
                label = f"🔧 {name}({args_preview})" if args_preview else f"🔧 {name}()"
                await self._send(session_id, start_tool_call(call.id, label, kind="execute", raw_input=args))
                tool = agent.tools.get(name)
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
                    # Check if tool requires confirmation
                    permission_granted = True
                    if self._requires_confirmation(name):
                        permission_granted = await self._request_tool_permission(
                            session_id=session_id,
                            tool_call_id=call.id,
                            tool_name=name,
                            tool_args=args,
                            tool_label=label,
                        )
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
                    else:
                        # No confirmation required, execute directly
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
    if config is None:
        if config_path is None:
            config = Config.load()
        else:
            config = Config.from_yaml(config_path)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    base_tools, skill_loader = await initialize_base_tools(config)
    prompt_path = config.find_config_file(config.agent.system_prompt_path)
    if prompt_path and prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = "You are a helpful AI assistant."
    if skill_loader:
        meta = skill_loader.get_skills_metadata_prompt()
        if meta:
            system_prompt = f"{system_prompt.rstrip()}\n\n{meta}"
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
