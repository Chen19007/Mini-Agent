#!/usr/bin/env python3
"""Interactive MCP Server - Local version with request_command_execution tool.

This server provides tools for requesting user confirmation and command execution.
It's a local fork of interactive-mcp with additional command execution support.
"""

import json
import sys
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback if FastMCP is not available
    print("Error: FastMCP not available. Please install: pip install mcp[fastmcp]", file=sys.stderr)
    sys.exit(1)

# Initialize the MCP server
mcp = FastMCP("interactive-mcp-local")


class RequestCommandExecutionInput(BaseModel):
    """Input model for request_command_execution tool."""

    command: str = Field(
        ...,
        description=(
            "The complete, executable command that includes: "
            "- Full absolute paths (not relative paths) "
            "- Required environment variables "
            "- Working directory context "
            "- All necessary parameters. "
            "The command should work correctly when executed from any directory by the user."
        ),
        min_length=1,
    )
    description: str = Field(
        ...,
        description="Clear description of what the command does and why it's needed.",
        min_length=1,
    )
    dangerous: bool = Field(
        default=False,
        description="Whether this is a dangerous operation (file deletion, system changes, etc.).",
    )
    working_directory: Optional[str] = Field(
        default=None,
        description="Optional working directory where the command should be executed.",
    )


@mcp.tool(
    name="request_command_execution",
    annotations={
        "title": "Request Command Execution",
        "readOnlyHint": False,
        "destructiveHint": True,  # May be destructive depending on command
        "idempotentHint": False,
        "openWorldHint": True,  # Interacts with external system
    },
)
async def request_command_execution(params: RequestCommandExecutionInput) -> str:
    """Request the user to execute a command and return the result.

    This tool is used when a command is not available as an MCP tool. It requests
    the user to execute the command manually and return the result.

    **CRITICAL**: The command must be complete and executable, including:
    - Full absolute paths (not relative paths)
    - Required environment variables
    - Working directory context
    - All necessary parameters

    The command should work correctly when executed from any directory by the user.

    Args:
        params (RequestCommandExecutionInput): Input parameters containing:
            - command (str): The complete, executable command
            - description (str): Clear description of what the command does
            - dangerous (bool): Whether this is a dangerous operation
            - working_directory (Optional[str]): Optional working directory

    Returns:
        str: A formatted message requesting the user to execute the command.
             The message includes the command, description, and instructions.
             The user should execute the command and return the result or decline.

    Examples:
        Good command (complete):
        ```
        cd /d "D:\\project\\my-app" && set PATH=%PATH%;D:\\nodejs && npm install
        ```

        Bad command (incomplete):
        ```
        npm install  # Missing directory context and environment setup
        ```
    """
    # Build the formatted message
    message_parts = []

    # Header
    if params.dangerous:
        message_parts.append("⚠️  DANGEROUS OPERATION REQUEST")
        message_parts.append("=" * 70)
    else:
        message_parts.append("📋 COMMAND EXECUTION REQUEST")
        message_parts.append("=" * 70)

    # Description
    message_parts.append(f"\nDescription: {params.description}")

    # Working directory
    if params.working_directory:
        message_parts.append(f"\nWorking Directory: {params.working_directory}")

    # Command
    message_parts.append(f"\nCommand to execute:")
    message_parts.append("-" * 70)
    message_parts.append(params.command)
    message_parts.append("-" * 70)

    # Instructions
    message_parts.append(
        "\n📝 Instructions:"
    )
    message_parts.append("1. Copy the command above")
    message_parts.append("2. Execute it in your terminal (in the specified working directory if provided)")
    message_parts.append("3. Return the output or 'DECLINED' if you choose not to execute it")

    if params.dangerous:
        message_parts.append("\n⚠️  WARNING: This is a dangerous operation. Please review the command carefully before executing.")

    message_parts.append("\n" + "=" * 70)

    return "\n".join(message_parts)


# For CLI mode, we can also provide a simple stdin/stdout interface
# But for MCP server, we just return the formatted message
# The agent will handle displaying it to the user

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()








