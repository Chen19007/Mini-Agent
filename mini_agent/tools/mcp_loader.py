"""MCP tool loader with real MCP client integration."""

import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .base import Tool, ToolResult


class MCPTool(Tool):
    """Wrapper for MCP tools."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        session: ClientSession,
    ):
        self._name = name
        self._description = description
        self._parameters = parameters
        self._session = session

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs) -> ToolResult:
        """Execute MCP tool via the session."""
        try:
            result = await self._session.call_tool(self._name, arguments=kwargs)

            # MCP tool results are a list of content items
            content_parts = []
            for item in result.content:
                if hasattr(item, 'text'):
                    content_parts.append(item.text)
                else:
                    content_parts.append(str(item))

            content_str = '\n'.join(content_parts)

            is_error = result.isError if hasattr(result, 'isError') else False

            # If there's an error, use the content as the error message
            # (MCP tools typically return error details in the content)
            if is_error:
                error_msg = content_str if content_str else "Tool returned error"
                return ToolResult(
                    success=False,
                    content="",
                    error=error_msg
                )
            else:
                return ToolResult(
                    success=True,
                    content=content_str,
                    error=None
                )
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"MCP tool execution failed: {str(e)}"
            )


class MCPServerConnection:
    """Manages connection to a single MCP server."""

    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack | None = None
        self.tools: list[MCPTool] = []

    async def connect(self) -> bool:
        """Connect to the MCP server using proper async context management."""
        try:
            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env if self.env else None
            )

            # Use AsyncExitStack to properly manage multiple async context managers
            self.exit_stack = AsyncExitStack()
            
            # Enter stdio client context
            read_stream, write_stream = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            
            # Enter client session context
            session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            self.session = session

            # Initialize the session
            await session.initialize()

            # List available tools
            tools_list = await session.list_tools()

            # Wrap each tool
            for tool in tools_list.tools:
                # Convert MCP tool schema to our format
                parameters = tool.inputSchema if hasattr(tool, 'inputSchema') else {}

                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=parameters,
                    session=session
                )
                self.tools.append(mcp_tool)

            print(f"✓ Connected to MCP server '{self.name}' - loaded {len(self.tools)} tools")
            for tool in self.tools:
                desc = tool.description[:60] if len(tool.description) > 60 else tool.description
                print(f"  - {tool.name}: {desc}...")
            return True

        except Exception as e:
            print(f"✗ Failed to connect to MCP server '{self.name}': {e}")
            # Clean up exit stack if connection failed
            if self.exit_stack:
                await self.exit_stack.aclose()
                self.exit_stack = None
            import traceback
            traceback.print_exc()
            return False

    async def disconnect(self):
        """Properly disconnect from the MCP server."""
        if self.exit_stack:
            # AsyncExitStack handles all cleanup properly
            await self.exit_stack.aclose()
            self.exit_stack = None
            self.session = None


# Global connections registry
_mcp_connections: list[MCPServerConnection] = []


async def _connect_server(
    server_name: str,
    command: str,
    args: list[str],
    env: dict[str, str] | None,
) -> tuple[MCPServerConnection, list[MCPTool]] | None:
    """
    Connect to a single MCP server.
    
    Args:
        server_name: Name of the server
        command: Command to run the server
        args: Command arguments
        env: Environment variables
        
    Returns:
        Tuple of (connection, tools) if successful, None otherwise
    """
    try:
        connection = MCPServerConnection(server_name, command, args, env)
        success = await connection.connect()
        
        if success:
            return (connection, connection.tools)
        return None
    except Exception as e:
        # Handle any unexpected exceptions (shouldn't happen, but be safe)
        print(f"✗ Unexpected error connecting to MCP server '{server_name}': {e}")
        import traceback
        traceback.print_exc()
        return None


async def load_mcp_tools_async(config_path: str = "mcp.json") -> list[Tool]:
    """
    Load MCP tools from config file.

    This function:
    1. Reads the MCP config file
    2. Starts MCP server processes
    3. Connects to each server
    4. Fetches tool definitions
    5. Wraps them as Tool objects

    Args:
        config_path: Path to MCP configuration file (default: "mcp.json")

    Returns:
        List of Tool objects representing MCP tools
    """
    global _mcp_connections

    config_file = Path(config_path)

    if not config_file.exists():
        print(f"MCP config not found: {config_path}")
        return []

    try:
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)

        mcp_servers = config.get("mcpServers", {})

        if not mcp_servers:
            print("No MCP servers configured")
            return []

        # Collect all enabled server configurations
        server_tasks = []
        for server_name, server_config in mcp_servers.items():
            if server_config.get("disabled", False):
                print(f"Skipping disabled server: {server_name}")
                continue

            command = server_config.get("command")
            args = server_config.get("args", [])
            env = server_config.get("env", {})

            if not command:
                print(f"No command specified for server: {server_name}")
                continue

            # Create connection task for parallel execution
            server_tasks.append(_connect_server(server_name, command, args, env))

        # Connect to all servers in parallel
        if server_tasks:
            print(f"Connecting to {len(server_tasks)} MCP server(s) in parallel...")
            results = await asyncio.gather(*server_tasks, return_exceptions=True)
        else:
            results = []

        # Process results
        all_tools = []
        for result in results:
            if isinstance(result, Exception):
                # Exception was already handled in _connect_server
                continue
            if result is not None:
                connection, tools = result
                _mcp_connections.append(connection)
                all_tools.extend(tools)

        print(f"\nTotal MCP tools loaded: {len(all_tools)}")

        return all_tools

    except Exception as e:
        print(f"Error loading MCP config: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_mcp_tools_metadata_prompt(mcp_tools: list[Tool]) -> str:
    """
    Generate a formatted prompt section describing available MCP tools.
    
    Args:
        mcp_tools: List of MCP Tool objects
        
    Returns:
        Formatted markdown string describing MCP tools
    """
    lines = []
    
    if not mcp_tools:
        lines.append("**Note**: No MCP tools are currently loaded. Configure MCP servers in `mcp.json` to enable additional tools.")
        return "\n".join(lines)
    
    lines.append("The following MCP tools are currently loaded and available:")
    lines.append("")
    
    for tool in mcp_tools:
        tool_name = tool.name
        tool_desc = tool.description or "No description available"
        
        # Format tool description
        lines.append(f"- **`{tool_name}`**: {tool_desc}")
        
        # Add parameter info if available
        if hasattr(tool, 'parameters') and tool.parameters:
            params = tool.parameters
            if isinstance(params, dict) and 'properties' in params:
                param_names = list(params['properties'].keys())
                if param_names:
                    # Format parameters nicely
                    param_list = ', '.join(f"`{name}`" for name in param_names[:5])  # Limit to 5 params for readability
                    if len(param_names) > 5:
                        param_list += f" (and {len(param_names) - 5} more)"
                    lines.append(f"  - Parameters: {param_list}")
    
    lines.append("")
    lines.append("**Usage**: Use these tools directly by name when you need their functionality.")
    
    return "\n".join(lines)


async def cleanup_mcp_connections():
    """Clean up all MCP connections."""
    global _mcp_connections
    for connection in _mcp_connections:
        await connection.disconnect()
    _mcp_connections.clear()
